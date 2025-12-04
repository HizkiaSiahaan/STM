"""
Real-time Remote Photoplethysmography (rPPG)
Sistem Teknologi Multimedia - Institut Teknologi Sumatera

NAMA: Hizkia Christovita Siahaan
NIM: 122140110
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
import matplotlib.pyplot as plt
import time

# =============================================================================
# KONFIGURASI
# =============================================================================
FPS = 30
WINDOW_SIZE = 10
BUFFER_SIZE = int(FPS * WINDOW_SIZE)

MIN_HR_BPM = 40
MAX_HR_BPM = 240
MIN_HR_HZ = MIN_HR_BPM / 60.0
MAX_HR_HZ = MAX_HR_BPM / 60.0

PLOT_UPDATE_INTERVAL = 30
BPM_SMOOTHING_WINDOW = 5

# Motion detection threshold
MOTION_THRESHOLD = 15.0

# Signal quality threshold (SNR)
MIN_SNR = 1.5

# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

def bandpass_filter(data, fps, low_freq, high_freq):
    """Apply Butterworth bandpass filter"""
    nyquist = fps / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are in valid range
    low = np.clip(low, 0.01, 0.99)
    high = np.clip(high, low + 0.01, 0.99)
    
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def calculate_snr(signal_data, fps):
    """Calculate Signal-to-Noise Ratio"""
    # Apply bandpass filter to get signal band
    filtered = bandpass_filter(signal_data, fps, MIN_HR_HZ, MAX_HR_HZ)
    
    # Signal power (in-band)
    signal_power = np.var(filtered)
    
    # Noise power (total - signal)
    total_power = np.var(signal_data)
    noise_power = total_power - signal_power
    
    if noise_power <= 0:
        return float('inf')
    
    snr = signal_power / noise_power
    return snr


def estimate_bpm_with_confidence(data, fps):
    """Estimate BPM using FFT with confidence score"""
    fft_data = fft(data)
    freqs = fftfreq(len(data), 1.0/fps)

    mask = (freqs >= MIN_HR_HZ) & (freqs <= MAX_HR_HZ)
    freqs = freqs[mask]
    power = np.abs(fft_data[mask])

    if len(power) == 0:
        return 0, 0, freqs, power

    # Find peak
    peak_idx = np.argmax(power)
    bpm = freqs[peak_idx] * 60.0
    
    # Calculate confidence (peak prominence)
    max_power = power[peak_idx]
    mean_power = np.mean(power)
    
    if mean_power > 0:
        confidence = (max_power - mean_power) / mean_power
    else:
        confidence = 0
    
    return bpm, confidence, freqs, power


def detect_motion(frame, prev_frame):
    """Detect motion between frames"""
    if prev_frame is None:
        return 0.0
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate frame difference
    diff = cv2.absdiff(gray1, gray2)
    motion_score = np.mean(diff)
    
    return motion_score


# =============================================================================
# FACE & ROI DETECTION
# =============================================================================

def get_forehead_roi(frame, face_landmarks):
    """Extract forehead ROI mask"""
    h, w = frame.shape[:2]

    # Get landmark points
    points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

    # Forehead landmarks
    roi_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
               397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
               172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    roi_points = [points[i] for i in roi_idx if i < len(points)]

    if len(roi_points) < 3:
        return None, None

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(roi_points, dtype=np.int32), 255)

    # Bounding box
    bbox = cv2.boundingRect(np.array(roi_points, dtype=np.int32))

    return mask, bbox


def get_cheeks_roi(frame, face_landmarks):
    """Extract cheeks ROI mask (additional ROI option)"""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
    
    # Left cheek landmarks
    left_cheek_idx = [123, 116, 100, 47, 126, 101, 50, 205, 207]
    # Right cheek landmarks  
    right_cheek_idx = [352, 345, 329, 277, 355, 330, 280, 425, 427]
    
    left_points = [points[i] for i in left_cheek_idx if i < len(points)]
    right_points = [points[i] for i in right_cheek_idx if i < len(points)]
    
    if len(left_points) < 3 or len(right_points) < 3:
        return None, None
    
    # Create combined mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(left_points, dtype=np.int32), 255)
    cv2.fillConvexPoly(mask, np.array(right_points, dtype=np.int32), 255)
    
    # Combined bounding box
    all_points = left_points + right_points
    bbox = cv2.boundingRect(np.array(all_points, dtype=np.int32))
    
    return mask, bbox


def extract_green_mean(frame, mask):
    """Extract mean green channel value from ROI"""
    if mask is None:
        return 0
    green = frame[:, :, 1]
    masked_values = green[mask > 0]
    
    if len(masked_values) == 0:
        return 0
    
    return np.mean(masked_values)


# =============================================================================
# rPPG PROCESSOR
# =============================================================================

class rPPGProcessor:
    def __init__(self, fps=30, window_size=10, smoothing=5):
        self.fps = fps
        self.buffer_size = int(fps * window_size)
        self.signal_buffer = deque(maxlen=self.buffer_size)
        
        # Enhanced BPM tracking with exponential moving average
        self.bpm_buffer = deque(maxlen=smoothing)
        self.current_bpm = 0
        self.bpm_ema = 0
        self.ema_alpha = 0.3  # EMA smoothing factor
        
        # Signal quality tracking
        self.snr = 0
        self.confidence = 0
        
        # Data for visualization
        self.filtered = None
        self.freqs = None
        self.power = None
        
        # Performance metrics
        self.process_times = deque(maxlen=30)

    def add_signal(self, value):
        """Add signal value to buffer"""
        self.signal_buffer.append(value)

    def process(self):
        """Process signal and estimate BPM"""
        if len(self.signal_buffer) < self.buffer_size:
            return False

        start_time = time.time()

        # Detrend and filter
        data = np.array(self.signal_buffer)
        data = data - np.mean(data)
        
        # Calculate SNR
        self.snr = calculate_snr(data, self.fps)
        
        # Apply bandpass filter
        self.filtered = bandpass_filter(data, self.fps, MIN_HR_HZ, MAX_HR_HZ)

        # Estimate BPM with confidence
        bpm, confidence, self.freqs, self.power = estimate_bpm_with_confidence(
            self.filtered, self.fps
        )
        
        self.confidence = confidence

        # Update BPM only if quality is good
        if 40 <= bpm <= 240 and self.snr >= MIN_SNR:
            self.bpm_buffer.append(bpm)
            
            # Update EMA
            if self.bpm_ema == 0:
                self.bpm_ema = bpm
            else:
                self.bpm_ema = self.ema_alpha * bpm + (1 - self.ema_alpha) * self.bpm_ema

        # Use EMA for current BPM
        if len(self.bpm_buffer) > 0:
            self.current_bpm = self.bpm_ema

        # Track processing time
        process_time = (time.time() - start_time) * 1000
        self.process_times.append(process_time)

        return True

    def get_bpm(self):
        """Get current BPM estimate"""
        return self.current_bpm

    def get_quality_metrics(self):
        """Get signal quality metrics"""
        return {
            'snr': self.snr,
            'confidence': self.confidence,
            'avg_process_time': np.mean(self.process_times) if self.process_times else 0
        }

    def get_data(self):
        """Get all data for visualization"""
        return {
            'raw': np.array(self.signal_buffer),
            'filtered': self.filtered,
            'freqs': self.freqs,
            'power': self.power
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

fig_plot = None
axes_plot = None
lines_plot = {}

def init_plot():
    """Initialize matplotlib figure for real-time plotting"""
    global fig_plot, axes_plot, lines_plot
    
    plt.ion()
    fig_plot, axes_plot = plt.subplots(3, 1, figsize=(10, 7))
    fig_plot.suptitle('rPPG Signal Analysis - Enhanced', fontsize=14, fontweight='bold')
    
    # Raw signal
    lines_plot['raw'], = axes_plot[0].plot([], [], 'b-', linewidth=1.5)
    axes_plot[0].set_title('Raw Signal (Green Channel)', fontsize=10)
    axes_plot[0].set_ylabel('Intensity')
    axes_plot[0].grid(True, alpha=0.3)
    
    # Filtered signal
    lines_plot['filtered'], = axes_plot[1].plot([], [], 'r-', linewidth=1.5)
    axes_plot[1].set_title('Filtered Signal (Bandpass 0.67-4.0 Hz)', fontsize=10)
    axes_plot[1].set_ylabel('Amplitude')
    axes_plot[1].set_xlabel('Samples')
    axes_plot[1].grid(True, alpha=0.3)
    
    # Frequency spectrum
    lines_plot['spectrum'], = axes_plot[2].plot([], [], 'g-', linewidth=1.5)
    lines_plot['peak'] = axes_plot[2].axvline(x=0, color='r', linestyle='--', 
                                               linewidth=2, visible=False, label='Peak')
    axes_plot[2].set_title('Frequency Spectrum', fontsize=10)
    axes_plot[2].set_xlabel('Heart Rate (BPM)')
    axes_plot[2].set_ylabel('Power')
    axes_plot[2].grid(True, alpha=0.3)
    axes_plot[2].set_xlim([MIN_HR_BPM, MAX_HR_BPM])
    axes_plot[2].legend(loc='upper right')
    
    plt.tight_layout()
    fig_plot.canvas.draw()
    fig_plot.canvas.flush_events()
    
    return fig_plot, axes_plot


def update_plot(data, bpm, quality_metrics):
    """Update matplotlib plots with enhanced information"""
    global fig_plot, axes_plot, lines_plot
    
    if fig_plot is None or not plt.fignum_exists(fig_plot.number):
        return
    
    try:
        # Update raw signal
        if data['raw'] is not None and len(data['raw']) > 0:
            x_raw = np.arange(len(data['raw']))
            lines_plot['raw'].set_data(x_raw, data['raw'])
            axes_plot[0].set_xlim(0, len(data['raw']))
            y_range = np.ptp(data['raw'])
            y_margin = max(5, y_range * 0.1)
            axes_plot[0].set_ylim(np.min(data['raw']) - y_margin, 
                                  np.max(data['raw']) + y_margin)
            
            # Add SNR to title
            snr = quality_metrics['snr']
            axes_plot[0].set_title(f'Raw Signal (Green Channel) - SNR: {snr:.2f}', 
                                   fontsize=10)
        
        # Update filtered signal
        if data['filtered'] is not None and len(data['filtered']) > 0:
            x_filt = np.arange(len(data['filtered']))
            lines_plot['filtered'].set_data(x_filt, data['filtered'])
            axes_plot[1].set_xlim(0, len(data['filtered']))
            y_min, y_max = np.min(data['filtered']), np.max(data['filtered'])
            margin = max(0.1, (y_max - y_min) * 0.15)
            axes_plot[1].set_ylim(y_min - margin, y_max + margin)
        
        # Update power spectrum
        if data['freqs'] is not None and data['power'] is not None and len(data['power']) > 0:
            bpm_freqs = data['freqs'] * 60
            lines_plot['spectrum'].set_data(bpm_freqs, data['power'])
            axes_plot[2].set_ylim(0, np.max(data['power']) * 1.15)
            
            # Update peak line and title with confidence
            if bpm > 0:
                lines_plot['peak'].set_xdata([bpm, bpm])
                lines_plot['peak'].set_visible(True)
                conf = quality_metrics['confidence']
                axes_plot[2].set_title(
                    f'Frequency Spectrum - BPM: {bpm:.1f} (Confidence: {conf:.2f})', 
                    fontsize=10
                )
            else:
                lines_plot['peak'].set_visible(False)
        
        fig_plot.canvas.draw_idle()
        fig_plot.canvas.flush_events()
        
    except Exception as e:
        pass


def draw_info(frame, bpm, buffer_pct, quality_metrics, motion_score, bbox=None):
    """Draw enhanced info overlay on frame"""
    
    # Draw ROI rectangle
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "ROI", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)

    # Main info box (larger for more info)
    box_height = 180
    cv2.rectangle(frame, (10, 10), (300, box_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, box_height), (255, 255, 255), 2)

    y_offset = 35
    
    # BPM with color coding
    if bpm > 0:
        if 60 <= bpm <= 100:
            color = (0, 255, 0)  # Green - normal
            status = "Normal"
        elif 40 <= bpm < 60 or 100 < bpm <= 120:
            color = (0, 255, 255)  # Yellow - borderline
            status = "Borderline"
        else:
            color = (0, 0, 255)  # Red - abnormal
            status = "Check"
        text = f"BPM: {bpm:.1f} ({status})"
    else:
        color = (128, 128, 128)
        text = "BPM: -- (Waiting)"
    
    cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
               0.65, color, 2)
    y_offset += 30

    # Buffer status
    buffer_color = (0, 255, 0) if buffer_pct >= 100 else (255, 255, 0)
    cv2.putText(frame, f"Buffer: {buffer_pct:.0f}%", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, buffer_color, 1)
    y_offset += 25

    # Signal quality (SNR)
    snr = quality_metrics['snr']
    snr_color = (0, 255, 0) if snr >= MIN_SNR else (0, 0, 255)
    snr_text = f"SNR: {snr:.2f}"
    if snr < MIN_SNR:
        snr_text += " (Low)"
    cv2.putText(frame, snr_text, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, snr_color, 1)
    y_offset += 25

    # Confidence
    conf = quality_metrics['confidence']
    conf_color = (0, 255, 0) if conf >= 1.0 else (255, 255, 0)
    cv2.putText(frame, f"Confidence: {conf:.2f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
    y_offset += 25

    # Motion warning
    motion_color = (0, 255, 0) if motion_score < MOTION_THRESHOLD else (0, 0, 255)
    motion_status = "Still" if motion_score < MOTION_THRESHOLD else "Motion!"
    cv2.putText(frame, f"Motion: {motion_status}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
    y_offset += 25

    # Processing time
    proc_time = quality_metrics['avg_process_time']
    cv2.putText(frame, f"Proc: {proc_time:.1f}ms", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Instructions at bottom
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# =============================================================================
# MAIN
# =============================================================================

def run_rppg():
    """Main function with enhanced features"""
    print("=" * 70)
    print("Real-time rPPG System - Enhanced Version")
    print("Sistem Teknologi Multimedia - ITERA")
    print("NAMA : Hizkia Christovita Siahaan")
    print("NIM: 122140110")
    print("=" * 70)
    print("\nENHANCEMENTS:")
    print("• Adaptive ROI selection (forehead)")
    print("• Signal quality indicator (SNR)")
    print("• Motion detection and warning")
    print("• Enhanced visualization with confidence level")
    print("• Exponential moving average for smoother BPM")
    print("• Real-time performance metrics")
    print("=" * 70)

    # Initialize MediaPipe
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize processor
    processor = rPPGProcessor(FPS, WINDOW_SIZE, BPM_SMOOTHING_WINDOW)

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return

    print("\nINSTRUCTIONS:")
    print("1. Position your face clearly in front of the camera")
    print("2. Stay still and wait for buffer to reach 100%")
    print("3. Keep motion indicator 'Still' for best results")
    print("4. Press 'q' to quit")
    print("=" * 70)

    # Setup matplotlib
    init_plot()

    frame_count = 0
    start_time = time.time()
    prev_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Motion detection
            motion_score = detect_motion(frame, prev_frame)
            prev_frame = frame.copy()

            # Detect face
            results = face_mesh.process(rgb_frame)
            bbox = None

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Use forehead ROI (you can switch to cheeks if needed)
                mask, bbox = get_forehead_roi(frame, landmarks)

                if mask is not None:
                    green_mean = extract_green_mean(frame, mask)
                    processor.add_signal(green_mean)

                    if len(processor.signal_buffer) >= processor.buffer_size:
                        processor.process()

            # Get metrics
            buffer_pct = (len(processor.signal_buffer) / processor.buffer_size) * 100
            bpm = processor.get_bpm()
            quality_metrics = processor.get_quality_metrics()

            # Draw info
            frame = draw_info(frame, bpm, buffer_pct, quality_metrics, 
                            motion_score, bbox)

            # Update plot periodically
            frame_count += 1
            if frame_count % PLOT_UPDATE_INTERVAL == 0 and buffer_pct >= 100:
                data = processor.get_data()
                update_plot(data, bpm, quality_metrics)

            # Show webcam
            cv2.imshow('rPPG Real-time - Enhanced', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close('all')
        face_mesh.close()

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 70)
        print("SESSION STATISTICS")
        print("=" * 70)
        print(f"Total Frames Processed: {frame_count}")
        print(f"Duration: {elapsed:.2f}s")
        print(f"Average FPS: {fps:.2f}")
        print(f"Final BPM Estimate: {bpm:.1f}")
        
        if processor.process_times:
            print(f"Avg Processing Time: {np.mean(processor.process_times):.2f}ms")
        
        print("=" * 70)
        print("Thank you for using rPPG Enhanced System!")
        print("=" * 70)


if __name__ == "__main__":
    run_rppg()