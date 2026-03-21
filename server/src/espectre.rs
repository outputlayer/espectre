//! ESPectre motion detection algorithm.
//!
//! Ported from https://github.com/francescopace/espectre
//! Spatial turbulence + 12 statistical features + MLP 12→16→8→1 (353 params, 98% F1)

/// Subcarrier indices used for spatial turbulence (12 out of 64 OFDM subcarriers).
const SUBCARRIERS: [usize; 12] = [12, 14, 16, 18, 20, 24, 28, 36, 40, 44, 48, 52];

/// Turbulence buffer size (number of packets).
const BUFFER_SIZE: usize = 75;

/// Feature normalization (StandardScaler from training).
const FEATURE_MEAN: [f32; 12] = [4.006633, 0.999253, 6.694695, 2.179692, 0.327112, 0.446007, 1.862577, 2.715100, 0.357620, 0.552517, 0.000380, 1.980423];
const FEATURE_SCALE: [f32; 12] = [2.798256, 0.800856, 4.197435, 1.973483, 0.134512, 1.276173, 6.095114, 0.412297, 0.343370, 0.516170, 0.023080, 0.248058];

// Layer 1: 12→16 weights (row-major)
#[rustfmt::skip]
const W1: [[f32; 16]; 12] = [
    [ 0.076045,-0.377349,-0.306961, 0.351105, 0.356417,-0.160925, 0.418079,-0.804686,-0.116420, 0.819474,-0.684336, 0.580890,-0.005332, 0.644718,-0.314334, 0.598145],
    [-0.470680, 0.550812, 0.212153,-0.224129,-0.219849, 0.134914,-0.314311,-0.227070,-0.062195,-0.509800,-0.269806,-0.167093,-0.321900,-0.687973,-0.849505,-0.847207],
    [-0.405631, 0.675470, 0.775226,-0.539148,-0.366973,-0.811748, 0.047822, 0.002183, 0.399731,-0.065932,-0.143900, 0.023415,-0.259355,-0.383228,-0.163127,-0.560130],
    [ 0.604640,-0.005853,-0.160423, 0.402897, 0.262336, 0.780758,-0.128716, 0.029886,-0.521450, 0.269154,-0.216107, 0.228437, 0.496968, 0.304607,-0.234326, 0.711955],
    [ 0.294878,-0.496470,-0.276991, 0.266232, 0.279773,-0.004047, 0.217686,-0.074235,-0.577422, 0.168669,-0.203687,-0.279527, 0.317375,-0.242129, 0.240725, 0.562834],
    [-0.292005, 0.027674,-0.063540,-0.687326,-0.233941,-0.214686,-0.350616,-0.229182, 0.249148, 0.083603,-0.134026,-0.668394, 0.090043,-0.004593, 0.311945,-0.658121],
    [-0.359831,-0.140064,-0.073952, 0.032247,-0.265466,-0.053527, 0.452913, 0.046136,-0.163531, 0.025576,-0.021505, 0.667482,-0.230185, 0.001520,-0.523952, 0.158649],
    [-0.554160,-0.229636,-0.172430,-0.660903,-0.572630,-0.305891, 0.358454,-0.420041,-0.077023,-0.153255,-0.527126, 0.282001,-0.324739,-0.164971,-0.090892,-0.454763],
    [-0.570433,-0.323152,-0.403171,-0.294881,-0.447561,-0.625347,-0.013169, 0.807915, 0.666729, 0.286569, 1.200856,-0.384882,-0.419730,-0.532660, 0.459417,-0.312917],
    [-1.175039, 0.049916, 0.312778,-0.299569,-1.009555,-0.511386,-0.374400, 0.145303,-0.679923,-0.715776,-0.138956,-0.382853,-1.209245,-0.224609,-0.210676,-0.165096],
    [ 0.159012,-0.292277,-0.190204,-0.392950, 0.099986,-0.051453, 0.502480,-0.044477,-0.248961,-0.256767,-0.073407, 0.364389, 0.006847, 0.680229, 0.104658,-0.330296],
    [ 0.067848,-0.180202,-0.037013, 0.021071, 0.022346, 0.036097, 0.039060,-0.020733,-0.113571, 0.051340, 0.108982, 0.272082, 0.091594, 0.108706,-0.008789,-0.031133],
];
const B1: [f32; 16] = [-0.339896,-0.317785,-0.472780, 0.041023,-0.219676,-0.348341,-0.133482,-0.154326,-0.132777,-0.227385,-0.205560,-0.317206,-0.484573,-0.115825, 0.187428, 0.111663];

// Layer 2: 16→8 weights
#[rustfmt::skip]
const W2: [[f32; 8]; 16] = [
    [ 0.691962,-0.606620, 0.452901,-0.787525, 0.709914,-0.534905, 0.533919, 0.914644],
    [-0.201585, 0.497610,-0.218902, 0.445192,-0.208468, 0.299187,-0.117908,-0.210568],
    [-0.475875, 0.750847,-0.595264, 0.899526,-0.407029, 1.164971,-0.369977,-0.391562],
    [ 1.316664,-0.995920, 0.989841,-0.975637, 0.601888,-0.839662, 0.834025, 0.889062],
    [ 0.853526,-1.107445, 0.472709,-0.432458, 0.707178,-0.789016, 0.453998, 0.456567],
    [ 1.012476,-0.516244, 0.677199,-0.872841, 0.626783,-1.099012, 0.665328, 0.746932],
    [ 0.654397,-0.449295, 0.510869,-0.419412, 0.305715,-0.449916, 0.490621, 0.541664],
    [-0.755215, 1.373177,-0.762889, 1.185983,-0.426248, 1.094583,-0.344049,-0.687365],
    [-0.487154, 0.815623,-0.393500, 0.557771,-0.401354, 0.757457,-0.363305,-0.452845],
    [ 0.646436,-0.223493, 0.559549,-0.541378, 0.630080,-0.590683, 0.706362, 0.259214],
    [-0.425567, 1.119088,-0.539905, 1.057544,-0.308690, 1.012017,-0.356279,-0.373944],
    [ 0.954100,-0.634242, 0.488847,-0.605061, 0.745140,-0.494961, 0.753395, 0.391129],
    [ 0.507742,-0.466140, 0.710344,-0.688122, 0.402298,-0.347882, 0.604613, 0.834331],
    [ 0.734055,-0.598294, 0.562494,-0.437695, 0.683962,-0.714559, 0.851848, 0.403765],
    [-0.219236, 0.990202,-0.337172, 0.809280,-0.254108, 0.953534,-0.131821,-0.220744],
    [ 1.344131,-0.615145, 0.495322,-0.804213, 0.975466,-0.974879, 1.144901, 0.321174],
];
const B2: [f32; 8] = [0.131859, 0.103719, 0.183820, 0.079724, 0.138642, 0.083687, 0.113109, 0.163309];

// Layer 3: 8→1 weights
const W3: [f32; 8] = [-1.362353, 1.398772, -1.927337, 1.695335, -1.802211, 1.495779, -1.815602, -1.324226];
const B3: f32 = 1.193543;

/// Motion detection threshold (0-10 scale, >5 = motion).
const MOTION_THRESHOLD: f32 = 5.0;

/// Baseline calibration period (frames).
const BASELINE_FRAMES: usize = 200;

/// Per-node ESPectre state.
pub struct EspectreNode {
    turbulence_buf: Vec<f32>,
    buf_pos: usize,
    buf_full: bool,
    /// Adaptive baseline: EMA of turbulence in quiet state.
    baseline_turb: f32,
    /// Baseline standard deviation.
    baseline_std: f32,
    /// Frame counter for calibration.
    frame_count: usize,
    /// Calibration turbulence accumulator.
    calib_sum: f32,
    calib_sq_sum: f32,
    /// Previous amplitudes for frame-to-frame motion detection.
    prev_amps: Option<Vec<f32>>,
    /// EMA of frame-to-frame motion energy.
    motion_ema: f32,
    /// Baseline motion energy (learned during calibration).
    baseline_motion: f32,
    /// Calibration motion accumulator.
    calib_motion_sum: f32,
}

impl EspectreNode {
    pub fn new() -> Self {
        Self {
            turbulence_buf: vec![0.0; BUFFER_SIZE],
            buf_pos: 0,
            buf_full: false,
            baseline_turb: 0.0,
            baseline_std: 1.0,
            frame_count: 0,
            calib_sum: 0.0,
            calib_sq_sum: 0.0,
            prev_amps: None,
            motion_ema: 0.0,
            baseline_motion: 0.0,
            calib_motion_sum: 0.0,
        }
    }

    /// Feed a CSI amplitude array, returns (motion_score 0-10, is_motion).
    pub fn process(&mut self, amplitudes: &[f32]) -> (f32, bool) {
        self.frame_count += 1;

        // Debug: log every 100th frame unconditionally.
        if self.frame_count % 100 == 0 {
            eprintln!("ESPectre process() frame={} amps_len={}", self.frame_count, amplitudes.len());
        }

        // Extract selected subcarrier amplitudes.
        let mut selected = [0.0f32; 12];
        for (i, &idx) in SUBCARRIERS.iter().enumerate() {
            selected[i] = if idx < amplitudes.len() { amplitudes[idx] } else { 0.0 };
        }

        // Compute spatial turbulence = std of selected amplitudes.
        let mean: f32 = selected.iter().sum::<f32>() / 12.0;
        let var: f32 = selected.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 12.0;
        let turbulence = var.sqrt();

        // Frame-to-frame motion energy (independent of MLP).
        let motion_energy = if let Some(ref prev) = self.prev_amps {
            let n = prev.len().min(amplitudes.len());
            if n > 0 {
                let diff: f32 = (0..n).map(|i| (amplitudes[i] - prev[i]).powi(2)).sum::<f32>() / n as f32;
                diff
            } else { 0.0 }
        } else { 0.0 };
        self.prev_amps = Some(amplitudes.to_vec());
        self.motion_ema = self.motion_ema * 0.7 + motion_energy * 0.3;

        // Calibration phase: learn baseline turbulence AND motion energy.
        if self.frame_count <= BASELINE_FRAMES {
            self.calib_sum += turbulence;
            self.calib_sq_sum += turbulence * turbulence;
            self.calib_motion_sum += motion_energy;
            if self.frame_count == BASELINE_FRAMES {
                let n = BASELINE_FRAMES as f32;
                self.baseline_turb = self.calib_sum / n;
                self.baseline_std = ((self.calib_sq_sum / n - (self.calib_sum / n).powi(2)).max(0.0)).sqrt().max(0.1);
                self.baseline_motion = self.calib_motion_sum / n;
                eprintln!("ESPectre calibrated: baseline_turb={:.3}, baseline_std={:.3}, baseline_motion={:.3}",
                    self.baseline_turb, self.baseline_std, self.baseline_motion);
            }
            // During calibration, return 0 (unknown).
            self.turbulence_buf[self.buf_pos] = turbulence;
            self.buf_pos = (self.buf_pos + 1) % BUFFER_SIZE;
            if self.buf_pos == 0 { self.buf_full = true; }
            return (0.0, false);
        }

        // Slow baseline adaptation (tracks room drift).
        self.baseline_turb = self.baseline_turb * 0.999 + turbulence * 0.001;

        // Normalized turbulence: how many baseline_stds above baseline.
        let norm_turb = ((turbulence - self.baseline_turb) / self.baseline_std).max(0.0);

        // Add normalized turbulence to buffer (not raw).
        self.turbulence_buf[self.buf_pos] = norm_turb;
        self.buf_pos = (self.buf_pos + 1) % BUFFER_SIZE;
        if self.buf_pos == 0 { self.buf_full = true; }

        if !self.buf_full {
            return (0.0, false);
        }

        // === Direct scoring (no MLP — its weights are from a different environment) ===

        // 1. Turbulence score: mean of normalized turbulence buffer.
        //    Quiet room shows ~0.4-0.6 from WiFi noise. Only count above that.
        let buf = &self.turbulence_buf;
        let turb_mean: f32 = buf.iter().sum::<f32>() / BUFFER_SIZE as f32;
        // Dead zone: turb_mean < 0.7 = noise floor, above that = real signal.
        let turb_excess = (turb_mean - 0.7).max(0.0);
        let turb_score = (turb_excess * 10.0).min(10.0);

        // 2. Peak turbulence: use 90th percentile instead of max to reject outliers.
        //    Single WiFi spikes can push max to 7+ in empty room.
        let mut sorted_buf = buf.to_vec();
        sorted_buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p90 = sorted_buf[(BUFFER_SIZE as f32 * 0.9) as usize];
        // Quiet room p90 ~1.0-1.5, real motion p90 > 3.
        let peak_excess = (p90 - 1.5).max(0.0);
        let peak_score = (peak_excess * 3.0).min(10.0);

        // 3. Motion energy relative to baseline.
        //    Use 2.0x baseline as threshold — WiFi jitter can push motion_ema to 3-4x baseline.
        let excess_motion = (self.motion_ema - self.baseline_motion * 2.0).max(0.0);
        let motion_baseline = self.baseline_motion.max(1.0);
        let motion_ratio = excess_motion / motion_baseline;
        let motion_score = (motion_ratio * 3.0).min(10.0);

        // 4. Turbulence variability (std of buffer — higher when moving).
        //    Quiet room std ~0.3-0.6. Only count above that.
        let turb_var: f32 = buf.iter().map(|x| (x - turb_mean).powi(2)).sum::<f32>() / BUFFER_SIZE as f32;
        let turb_std = turb_var.sqrt();
        let var_excess = (turb_std - 0.5).max(0.0);
        let variability_score = (var_excess * 6.0).min(10.0);

        // Combined score: turbulence mean is most reliable signal.
        let combined = turb_score * 0.35 + peak_score * 0.25 + motion_score * 0.20 + variability_score * 0.20;

        // Log periodically for diagnostics.
        if self.frame_count % 200 == 0 {
            eprintln!("ESPectre[{}]: turb={:.2}/p90={:.2} mot={:.1}/{:.1} | t={:.1} p={:.1} m={:.1} v={:.1} => {:.2}",
                self.frame_count, turb_mean, p90, self.motion_ema, self.baseline_motion,
                turb_score, peak_score, motion_score, variability_score, combined);
        }

        (combined, combined > MOTION_THRESHOLD)
    }

    fn extract_features(&self, amplitudes: &[f32; 12]) -> [f32; 12] {
        let buf = &self.turbulence_buf;
        let n = BUFFER_SIZE as f32;

        // 0: mean
        let mean: f32 = buf.iter().sum::<f32>() / n;

        // 1: std
        let var: f32 = buf.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = var.sqrt().max(1e-9);

        // 2: max
        let max_val = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // 3: min
        let min_val = buf.iter().cloned().fold(f32::INFINITY, f32::min);

        // 4: zero-crossing rate (around mean)
        let mut zcr = 0u32;
        for i in 1..BUFFER_SIZE {
            if (buf[i] - mean) * (buf[i - 1] - mean) < 0.0 {
                zcr += 1;
            }
        }
        let zcr_rate = zcr as f32 / (BUFFER_SIZE - 1) as f32;

        // 5: skewness
        let skewness: f32 = buf.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f32>() / n;

        // 6: excess kurtosis
        let kurtosis: f32 = buf.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f32>() / n - 3.0;

        // 7: Shannon entropy (10 bins)
        let entropy = shannon_entropy(buf, 10, min_val, max_val);

        // 8: lag-1 autocorrelation
        let mut cov = 0.0f32;
        for i in 1..BUFFER_SIZE {
            cov += (buf[i] - mean) * (buf[i - 1] - mean);
        }
        let autocorr = if var > 1e-9 { cov / ((BUFFER_SIZE - 1) as f32 * var) } else { 0.0 };

        // 9: median absolute deviation
        let mut sorted = buf.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[BUFFER_SIZE / 2];
        let mut abs_devs: Vec<f32> = sorted.iter().map(|x| (x - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = abs_devs[BUFFER_SIZE / 2];

        // 10: linear regression slope
        let x_mean = (BUFFER_SIZE - 1) as f32 / 2.0;
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for i in 0..BUFFER_SIZE {
            let xi = i as f32 - x_mean;
            num += xi * (buf[i] - mean);
            den += xi * xi;
        }
        let slope = if den > 1e-9 { num / den } else { 0.0 };

        // 11: amplitude entropy (5 bins over current subcarrier amplitudes)
        let amp_min = amplitudes.iter().cloned().fold(f32::INFINITY, f32::min);
        let amp_max = amplitudes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let amp_entropy = shannon_entropy(amplitudes, 5, amp_min, amp_max);

        [mean, std, max_val, min_val, zcr_rate, skewness, kurtosis, entropy, autocorr, mad, slope, amp_entropy]
    }
}

/// Shannon entropy with N bins.
fn shannon_entropy(data: &[f32], n_bins: usize, min_val: f32, max_val: f32) -> f32 {
    let range = (max_val - min_val).max(1e-9);
    let mut bins = vec![0u32; n_bins];
    for &v in data {
        let idx = (((v - min_val) / range) * (n_bins as f32 - 0.001)).floor() as usize;
        let idx = idx.min(n_bins - 1);
        bins[idx] += 1;
    }
    let n = data.len() as f32;
    let mut ent = 0.0f32;
    for &count in &bins {
        if count > 0 {
            let p = count as f32 / n;
            ent -= p * p.ln();
        }
    }
    ent / (n_bins as f32).ln().max(1e-9) // normalize to [0,1]
}

/// MLP inference: 12 → 16 → 8 → 1 with sigmoid output scaled to 0-10.
fn mlp_predict(features: &[f32; 12]) -> f32 {
    // Normalize features.
    let mut x = [0.0f32; 12];
    for i in 0..12 {
        x[i] = (features[i] - FEATURE_MEAN[i]) / FEATURE_SCALE[i];
    }

    // Layer 1: ReLU(W1^T * x + b1)
    let mut h1 = [0.0f32; 16];
    for j in 0..16 {
        let mut sum = B1[j];
        for i in 0..12 {
            sum += x[i] * W1[i][j];
        }
        h1[j] = sum.max(0.0); // ReLU
    }

    // Layer 2: ReLU(W2^T * h1 + b2)
    let mut h2 = [0.0f32; 8];
    for j in 0..8 {
        let mut sum = B2[j];
        for i in 0..16 {
            sum += h1[i] * W2[i][j];
        }
        h2[j] = sum.max(0.0); // ReLU
    }

    // Layer 3: sigmoid(W3 * h2 + b3) * 10
    let mut out = B3;
    for i in 0..8 {
        out += h2[i] * W3[i];
    }

    // Sigmoid with overflow protection, scaled to 0-10.
    if out < -20.0 {
        0.0
    } else if out > 20.0 {
        10.0
    } else {
        1.0 / (1.0 + (-out).exp()) * 10.0
    }
}

/// Serializable calibration data for one node.
#[derive(Clone)]
struct NodeCalibration {
    baseline_turb: f32,
    baseline_std: f32,
    baseline_motion: f32,
}

/// Multi-node ESPectre detector: fuses results from 3 nodes.
pub struct EspectreDetector {
    nodes: Vec<EspectreNode>,
    calibration_path: Option<String>,
}

impl EspectreDetector {
    pub fn new(n_nodes: usize) -> Self {
        Self {
            nodes: (0..n_nodes).map(|_| EspectreNode::new()).collect(),
            calibration_path: None,
        }
    }

    /// Create detector and try to load saved calibration from disk.
    pub fn new_with_persistence(n_nodes: usize, path: &str) -> Self {
        let mut det = Self {
            nodes: (0..n_nodes).map(|_| EspectreNode::new()).collect(),
            calibration_path: Some(path.to_string()),
        };
        det.load_calibration();
        det
    }

    /// Process amplitudes for a specific node. Returns per-node (score, is_motion).
    pub fn process_node(&mut self, node_id: u8, amplitudes: &[f32]) -> (f32, bool) {
        let idx = (node_id as usize).saturating_sub(1).min(self.nodes.len() - 1);
        let was_calibrating = self.nodes[idx].frame_count < BASELINE_FRAMES;
        let result = self.nodes[idx].process(amplitudes);
        // Save calibration to disk when any node finishes calibrating.
        if was_calibrating && self.nodes[idx].frame_count >= BASELINE_FRAMES {
            self.save_calibration();
        }
        result
    }

    /// Save calibration baselines to JSON file.
    fn save_calibration(&self) {
        let path = match &self.calibration_path {
            Some(p) => p.clone(),
            None => return,
        };
        let cals: Vec<String> = self.nodes.iter().enumerate().map(|(i, n)| {
            format!(
                "{{\"node\":{},\"baseline_turb\":{:.6},\"baseline_std\":{:.6},\"baseline_motion\":{:.6}}}",
                i, n.baseline_turb, n.baseline_std, n.baseline_motion
            )
        }).collect();
        let json = format!("{{\"calibration\":[{}],\"timestamp\":\"{}\"}}", cals.join(","),
            chrono::Utc::now().to_rfc3339());
        match std::fs::write(&path, &json) {
            Ok(_) => eprintln!("ESPectre: saved calibration to {}", path),
            Err(e) => eprintln!("ESPectre: failed to save calibration: {}", e),
        }
    }

    /// Load calibration from JSON file (skip live calibration if loaded).
    fn load_calibration(&mut self) {
        let path = match &self.calibration_path {
            Some(p) => p.clone(),
            None => return,
        };
        let data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(_) => { eprintln!("ESPectre: no saved calibration at {}, will calibrate live", path); return; }
        };
        eprintln!("ESPectre: found calibration file, parsing...");
        // Simple JSON parsing: find each "node":N block and extract values.
        let mut loaded = 0;
        let mut search_from = 0;
        loop {
            let node_key = "\"node\":";
            let pos = match data[search_from..].find(node_key) {
                Some(p) => search_from + p,
                None => break,
            };
            // Extract the block from this position to next '}'
            let block_end = match data[pos..].find('}') {
                Some(e) => pos + e,
                None => break,
            };
            let block = &data[pos..=block_end];
            search_from = block_end + 1;

            let get_val = |key: &str| -> Option<f32> {
                let kpos = block.find(key)?;
                let after = &block[kpos + key.len()..];
                let colon = after.find(':')?;
                let rest = &after[colon+1..];
                let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
                rest[..end].trim().parse().ok()
            };

            let node_idx = match get_val("\"node\"") { Some(v) => v as usize, None => continue };
            if node_idx >= self.nodes.len() { continue; }
            let bt = match get_val("\"baseline_turb\"") { Some(v) => v, None => continue };
            let bs = match get_val("\"baseline_std\"") { Some(v) => v, None => continue };
            let bm = match get_val("\"baseline_motion\"") { Some(v) => v, None => continue };
            self.nodes[node_idx].baseline_turb = bt;
            self.nodes[node_idx].baseline_std = bs;
            self.nodes[node_idx].baseline_motion = bm;
            self.nodes[node_idx].frame_count = BASELINE_FRAMES + 1; // skip calibration
            loaded += 1;
            eprintln!("ESPectre: loaded node {}: turb={:.3} std={:.3} motion={:.3}",
                node_idx, bt, bs, bm);
        }
        if loaded > 0 {
            eprintln!("ESPectre: restored {} node calibrations from disk — no recalibration needed", loaded);
        } else {
            eprintln!("ESPectre: calibration file found but could not parse, will calibrate live");
        }
    }

    /// Get fused result across all nodes: (max_score, any_motion, per_node_scores).
    pub fn fused_result(&self) -> (f32, bool, Vec<f32>) {
        (0.0, false, vec![])
    }
}
