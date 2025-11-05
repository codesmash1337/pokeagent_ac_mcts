//! Neural evaluation helpers that delegate to the Python `NeuralInferenceRunner`.

use crate::state::{SideReference, State};

#[cfg(feature = "neural")]
use once_cell::sync::OnceCell;
#[cfg(feature = "neural")]
use pyo3::{
    exceptions::PyKeyError,
    prelude::*,
    types::{PyDict, PyList},
};
const VERBOSE_NEURAL_EVAL: bool = true;

macro_rules! verbose_eval {
    ($($arg:tt)*) => {
        if VERBOSE_NEURAL_EVAL {
            eprintln!($($arg)*);
        }
    };
}

#[derive(Clone)]
pub struct NeuralEvaluation {
    pub value: f32,  // Normalized to [-1, 1]
    pub raw_value: f32,  // Raw critic output (typically [-1100, 1100])
    pub policy: Vec<f32>,
}

/// Returns a value estimate in `[-1, 1]` for a given state. When the `neural` feature
/// is disabled, this function always returns `None` so callers can fall back to the
/// hand-crafted evaluation function.
pub fn neural_state_value(state: &State) -> Option<NeuralEvaluation> {
    #[cfg(feature = "neural")]
    {
        let states = [state];
        match python_state_values_batch(&states) {
            Ok(mut evals) => evals.into_iter().next(),
            Err(err) => {
                let message = err.to_string();
                Python::with_gil(|py| err.print(py));
                verbose_eval!("neural evaluation failed: {message}");
                None
            }
        }
    }
    #[cfg(not(feature = "neural"))]
    {
        let _ = state; // silence warnings
        verbose_eval!(
            "neural feature disabled: enable \"neural\" feature to use the critic evaluator"
        );
        None
    }
}

pub fn neural_state_values_batch(states: &[&State]) -> Option<Vec<NeuralEvaluation>> {
    #[cfg(feature = "neural")]
    {
        if states.is_empty() {
            return Some(Vec::new());
        }
        match python_state_values_batch(states) {
            Ok(values) => Some(values),
            Err(err) => {
                let message = err.to_string();
                Python::with_gil(|py| err.print(py));
                verbose_eval!("batched neural evaluation failed: {message}");
                None
            }
        }
    }
    #[cfg(not(feature = "neural"))]
    {
        let _ = states;
        verbose_eval!(
            "neural feature disabled: enable \"neural\" feature to use the critic evaluator"
        );
        None
    }
}

/// Perspective-aware single-state evaluation. Returns policy/value from the given side's
/// perspective ("side_one" or "side_two").
pub fn neural_state_value_for_side(state: &State, side: SideReference) -> Option<NeuralEvaluation> {
    #[cfg(feature = "neural")]
    {
        let states = [state];
        match python_state_values_batch_with_perspective(&states, side) {
            Ok(mut evals) => evals.into_iter().next(),
            Err(err) => {
                let message = err.to_string();
                Python::with_gil(|py| err.print(py));
                verbose_eval!("neural evaluation (perspective) failed: {message}");
                None
            }
        }
    }
    #[cfg(not(feature = "neural"))]
    {
        let _ = (state, side);
        verbose_eval!(
            "neural feature disabled: enable \"neural\" feature to use the critic evaluator"
        );
        None
    }
}

/// Perspective-aware batched evaluation. Each state's policy/value is computed from the provided side.
pub fn neural_state_values_batch_for_side(
    states: &[&State],
    side: SideReference,
) -> Option<Vec<NeuralEvaluation>> {
    #[cfg(feature = "neural")]
    {
        if states.is_empty() {
            return Some(Vec::new());
        }
        match python_state_values_batch_with_perspective(states, side) {
            Ok(values) => Some(values),
            Err(err) => {
                let message = err.to_string();
                Python::with_gil(|py| err.print(py));
                verbose_eval!("batched neural evaluation (perspective) failed: {message}");
                None
            }
        }
    }
    #[cfg(not(feature = "neural"))]
    {
        let _ = (states, side);
        verbose_eval!(
            "neural feature disabled: enable \"neural\" feature to use the critic evaluator"
        );
        None
    }
}

/// Decode token IDs to their string representations
#[cfg(feature = "neural")]
fn decode_tokens(py: Python<'_>, token_ids: &[i32]) -> Option<Vec<String>> {
    match py.import("poke_engine") {
        Ok(module) => {
            match module.getattr("get_tokenizer") {
                Ok(get_tokenizer_fn) => {
                    match get_tokenizer_fn.call0() {
                        Ok(tokenizer) => {
                            match tokenizer.getattr("decode") {
                                Ok(decode_fn) => {
                                    let decoded_tokens: Result<Vec<String>, _> = token_ids
                                        .iter()
                                        .map(|&token_id| {
                                            decode_fn.call1((token_id,))
                                                .and_then(|result| result.extract::<String>())
                                        })
                                        .collect();
                                    decoded_tokens.ok()
                                }
                                Err(_e) => {
                                    // eprintln!("[TOKEN DECODE ERROR] Failed to get decode method: {}", e);
                                    None
                                }
                            }
                        }
                        Err(_e) => {
                            // eprintln!("[TOKEN DECODE ERROR] Failed to call get_tokenizer: {}", e);
                            None
                        }
                    }
                }
                Err(_e) => {
                    // eprintln!("[TOKEN DECODE ERROR] Failed to get get_tokenizer function: {}", e);
                    None
                }
            }
        }
        Err(_e) => {
            // eprintln!("[TOKEN DECODE ERROR] Failed to import poke_engine: {}", e);
            None
        }
    }
}

/// Get observation (text tokens and numerical features) for logging purposes
pub fn get_observation_for_logging(state: &State, side: SideReference) -> Option<(Vec<i32>, Vec<f32>, Vec<String>)> {
    #[cfg(feature = "neural")]
    {
        Python::with_gil(|py| {
            let perspective = match side {
                SideReference::SideOne => "side_one",
                SideReference::SideTwo => "side_two",
            };
            
            match py.import("poke_engine") {
                Ok(module) => {
                    match state_to_python(py, state) {
                        Ok(py_state) => {
                            match module.getattr("prepare_inference_payload") {
                                Ok(prepare_fn) => {
                                    match prepare_fn.call1((py_state, perspective, battle_format())) {
                                        Ok(payload) => {
                                            // Try to access as dictionary first
                                            if let Ok(dict) = payload.downcast::<pyo3::types::PyDict>() {
                                                match (dict.get_item("text_tokens"), dict.get_item("numbers")) {
                                                    (Ok(Some(text_tokens_obj)), Ok(Some(numbers_obj))) => {
                                                        match (text_tokens_obj.extract::<Vec<i32>>(), numbers_obj.extract::<Vec<f32>>()) {
                                                            (Ok(text_tokens), Ok(numbers)) => {
                                                                let decoded = decode_tokens(py, &text_tokens).unwrap_or_default();
                                                                Some((text_tokens, numbers, decoded))
                                                            }
                                                            (Err(e1), _) => {
                                                                eprintln!("[OBSERVATION ERROR] Failed to extract text_tokens: {}", e1);
                                                                None
                                                            }
                                                            (_, Err(e2)) => {
                                                                eprintln!("[OBSERVATION ERROR] Failed to extract numbers: {}", e2);
                                                                None
                                                            }
                                                        }
                                                    }
                                                    (Err(e), _) => {
                                                        eprintln!("[OBSERVATION ERROR] Failed to get text_tokens from dict: {}", e);
                                                        None
                                                    }
                                                    (_, Err(e)) => {
                                                        eprintln!("[OBSERVATION ERROR] Failed to get numbers from dict: {}", e);
                                                        None
                                                    }
                                                    (Ok(None), _) => {
                                                        eprintln!("[OBSERVATION ERROR] text_tokens key not found in dict");
                                                        None
                                                    }
                                                    (_, Ok(None)) => {
                                                        eprintln!("[OBSERVATION ERROR] numbers key not found in dict");
                                                        None
                                                    }
                                                }
                                            } else {
                                                eprintln!("[OBSERVATION ERROR] Payload is not a dictionary");
                                                None
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("[OBSERVATION ERROR] Failed to call prepare_inference_payload: {}", e);
                                            e.print(py);
                                            None
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("[OBSERVATION ERROR] Failed to get prepare_inference_payload attribute: {}", e);
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[OBSERVATION ERROR] Failed to convert state to Python: {}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[OBSERVATION ERROR] Failed to import poke_engine module: {}", e);
                    None
                }
            }
        })
    }
    #[cfg(not(feature = "neural"))]
    {
        let _ = (state, side);
        eprintln!("[OBSERVATION ERROR] Neural feature not enabled");
        None
    }
}

/// Get observation without decoding for compatibility
pub fn get_observation_for_logging_no_decode(state: &State, side: SideReference) -> Option<(Vec<i32>, Vec<f32>)> {
    get_observation_for_logging(state, side).map(|(tokens, numbers, _)| (tokens, numbers))
}

// Min-max normalization to [-1, 1]: Consistent with terminal states
// Preserves linear differences and matches terminal state range
#[cfg(feature = "neural")]
fn normalize_to_advantage_range(raw: f32) -> f32 {
    let min = -1100.0;
    let max = 1100.0;
    // Map [min, max] -> [-1, 1]
    // First map to [0, 1]: (raw - min) / (max - min)
    // Then map to [-1, 1]: 2 * x - 1
    let normalized = (raw - min) / (max - min);
    (2.0 * normalized - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
}


#[cfg(feature = "neural")]
pub fn python_state_values_batch(states: &[&State]) -> PyResult<Vec<NeuralEvaluation>> {
    Python::with_gil(|py| {
        let runner = get_runner(py)?;
        runner.as_ref(py).call_method0("reset")?;
        let pointers = state_pointers(states);
        let py_pointers: Py<PyList> = PyList::new(py, &pointers).into();

        let module = py.import("poke_engine")?;
        let prepare_fn = module.getattr("prepare_inference_payload_batch_from_pointers")?;
        let payload = prepare_fn.call1((
            py_pointers.as_ref(py),
            "side_one",
            battle_format(),
        ))?;
        let payload_dict = payload.downcast::<PyDict>()?;

        let text_tokens = payload_dict
            .get_item("text_tokens")?
            .ok_or_else(|| PyKeyError::new_err("text_tokens"))?
            .to_object(py);
        let numbers = payload_dict
            .get_item("numbers")?
            .ok_or_else(|| PyKeyError::new_err("numbers"))?
            .to_object(py);
        let legal_actions = payload_dict
            .get_item("legal_actions")?
            .ok_or_else(|| PyKeyError::new_err("legal_actions"))?
            .to_object(py);
        let move_mappings = payload_dict
            .get_item("move_mappings")?
            .ok_or_else(|| PyKeyError::new_err("move_mappings"))?
            .to_object(py);
        let switch_mappings = payload_dict
            .get_item("switch_mappings")?
            .ok_or_else(|| PyKeyError::new_err("switch_mappings"))?
            .to_object(py);

        let kwargs = PyDict::new(py);
        kwargs.set_item("battle_format", battle_format())?;

        let results = runner.as_ref(py).call_method(
            "infer_from_payload_batch",
            (
                text_tokens,
                numbers,
                legal_actions,
                move_mappings,
                switch_mappings,
            ),
            Some(kwargs),
        )?;
        let result_list = results.downcast::<PyList>()?;
        let mut evals = Vec::with_capacity(result_list.len());
        for item in result_list.iter() {
            let state_value = item.getattr("state_value")?;
            let raw_value: f32 = state_value.call_method0("item")?.extract()?;
            // eprintln!("[CRITIC RAW OUTPUT] raw_value from critic: {:.6}", raw_value);
            record_raw_value_sample(raw_value);
            let mut value = raw_value;
            value = normalize_to_advantage_range(value);
            // eprintln!("[CRITIC NORMALIZED] normalized to [-1,1]: {:.6}", value);
            let policy_prior = item.getattr("policy_prior")?;
            let policy: Vec<f32> = policy_prior.extract()?;
            evals.push(NeuralEvaluation { value, raw_value, policy });
        }
        Ok(evals)
    })
}

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Mutex, OnceLock};

const LOG_VALUE_SCALE_SAMPLES: bool = true;

static RAW_VALUE_SAMPLES: OnceLock<Mutex<Vec<f32>>> = OnceLock::new();

fn record_raw_value_sample(raw: f32) {
    if !LOG_VALUE_SCALE_SAMPLES {
        return;
    }
    let container = RAW_VALUE_SAMPLES.get_or_init(|| Mutex::new(Vec::with_capacity(1024)));
    if let Ok(mut guard) = container.lock() {
        guard.push(raw);
    }
}

pub fn drain_raw_value_samples() -> Vec<f32> {
    if !LOG_VALUE_SCALE_SAMPLES {
        return Vec::new();
    }
    let Some(container) = RAW_VALUE_SAMPLES.get() else {
        return Vec::new();
    };
    if let Ok(mut guard) = container.lock() {
        let mut out = Vec::new();
        std::mem::swap(&mut *guard, &mut out);
        out
    } else {
        Vec::new()
    }
}

static PYTHON_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
static PYTHON_CALL_TOTAL_TIME: AtomicU64 = AtomicU64::new(0);
static PYTHON_CALL_GIL_TIME: AtomicU64 = AtomicU64::new(0);
static PYTHON_CALL_PREP_TIME: AtomicU64 = AtomicU64::new(0);
static PYTHON_CALL_INFER_TIME: AtomicU64 = AtomicU64::new(0);
static PYTHON_CALL_EXTRACT_TIME: AtomicU64 = AtomicU64::new(0);

pub fn reset_python_call_stats() {
    PYTHON_CALL_COUNT.store(0, AtomicOrdering::Relaxed);
    PYTHON_CALL_TOTAL_TIME.store(0, AtomicOrdering::Relaxed);
    PYTHON_CALL_GIL_TIME.store(0, AtomicOrdering::Relaxed);
    PYTHON_CALL_PREP_TIME.store(0, AtomicOrdering::Relaxed);
    PYTHON_CALL_INFER_TIME.store(0, AtomicOrdering::Relaxed);
    PYTHON_CALL_EXTRACT_TIME.store(0, AtomicOrdering::Relaxed);
    
    // Also reset Python-side MODEL_TIMING stats
    #[cfg(feature = "neural")]
    {
        let _ = Python::with_gil(|py| {
            if let Ok(module) = py.import("poke_engine.neural_runner") {
                let _ = module.call_method0("reset_model_timing_stats");
            }
        });
    }
}

pub fn log_python_call_stats() {
    let count = PYTHON_CALL_COUNT.load(AtomicOrdering::Relaxed);
    if count == 0 {
        return;
    }
    let total = PYTHON_CALL_TOTAL_TIME.load(AtomicOrdering::Relaxed) as f64 / 1000.0;
    let gil = PYTHON_CALL_GIL_TIME.load(AtomicOrdering::Relaxed) as f64 / 1000.0;
    let prep = PYTHON_CALL_PREP_TIME.load(AtomicOrdering::Relaxed) as f64 / 1000.0;
    let infer = PYTHON_CALL_INFER_TIME.load(AtomicOrdering::Relaxed) as f64 / 1000.0;
    let extract = PYTHON_CALL_EXTRACT_TIME.load(AtomicOrdering::Relaxed) as f64 / 1000.0;
    
    eprintln!("[PYTHON_CALL] batches={} gil={:.1}ms prep={:.1}ms infer={:.1}ms extract={:.1}ms total={:.1}ms", 
        count, gil, prep, infer, extract, total);
    
    // Also log Python-side MODEL_TIMING stats
    #[cfg(feature = "neural")]
    {
        let _ = Python::with_gil(|py| {
            if let Ok(module) = py.import("poke_engine.neural_runner") {
                let _ = module.call_method0("log_model_timing_stats");
            }
        });
    }
}

#[cfg(feature = "neural")]
pub fn python_state_values_batch_with_perspective(
    states: &[&State],
    side: SideReference,
) -> PyResult<Vec<NeuralEvaluation>> {
    let rust_start = std::time::Instant::now();
    
    let result = Python::with_gil(|py| {
        let gil_time = rust_start.elapsed().as_secs_f64() * 1000.0;
        
        let prep_start = std::time::Instant::now();
        let runner = get_runner(py)?;
        runner.as_ref(py).call_method0("reset")?;
        let pointers = state_pointers(states);
        let py_pointers: Py<PyList> = PyList::new(py, &pointers).into();
        let perspective = match side {
            SideReference::SideOne => "side_one",
            SideReference::SideTwo => "side_two",
        };

        let module = py.import("poke_engine")?;
        let prepare_fn = module.getattr("prepare_inference_payload_batch_from_pointers")?;
        let payload = prepare_fn.call1((
            py_pointers.as_ref(py),
            perspective,
            battle_format(),
        ))?;
        let payload_dict = payload.downcast::<PyDict>()?;

        let text_tokens = payload_dict
            .get_item("text_tokens")?
            .ok_or_else(|| PyKeyError::new_err("text_tokens"))?
            .to_object(py);
        let numbers = payload_dict
            .get_item("numbers")?
            .ok_or_else(|| PyKeyError::new_err("numbers"))?
            .to_object(py);
        let legal_actions = payload_dict
            .get_item("legal_actions")?
            .ok_or_else(|| PyKeyError::new_err("legal_actions"))?
            .to_object(py);
        let move_mappings = payload_dict
            .get_item("move_mappings")?
            .ok_or_else(|| PyKeyError::new_err("move_mappings"))?
            .to_object(py);
        let switch_mappings = payload_dict
            .get_item("switch_mappings")?
            .ok_or_else(|| PyKeyError::new_err("switch_mappings"))?
            .to_object(py);
        let prep_time = prep_start.elapsed().as_secs_f64() * 1000.0;

        let kwargs = PyDict::new(py);
        kwargs.set_item("battle_format", battle_format())?;


        let infer_start = std::time::Instant::now();
        let results = runner.as_ref(py).call_method(
            "infer_from_payload_batch",
            (
                text_tokens,
                numbers,
                legal_actions,
                move_mappings,
                switch_mappings,
            ),
            Some(kwargs),
        )?;
        let infer_time = infer_start.elapsed().as_secs_f64() * 1000.0;
        
        let extract_start = std::time::Instant::now();
        let result_list = results.downcast::<PyList>()?;
        let mut evals = Vec::with_capacity(result_list.len());
        for item in result_list.iter() {
            let state_value = item.getattr("state_value")?;
            let raw_value: f32 = state_value.call_method0("item")?.extract()?;
            // eprintln!("[CRITIC RAW OUTPUT] raw_value from critic: {:.6}", raw_value);
            record_raw_value_sample(raw_value);
            let mut value = raw_value;
            value = normalize_to_advantage_range(value);
            // eprintln!("[CRITIC NORMALIZED] normalized to [-1,1]: {:.6}", value);
            let policy_prior = item.getattr("policy_prior")?;
            let policy: Vec<f32> = policy_prior.extract()?;
            evals.push(NeuralEvaluation { value, raw_value, policy });
        }
        let extract_time = extract_start.elapsed().as_secs_f64() * 1000.0;
        
        let total_time = rust_start.elapsed().as_secs_f64() * 1000.0;
        
        // Accumulate timing stats
        PYTHON_CALL_COUNT.fetch_add(1, AtomicOrdering::Relaxed);
        PYTHON_CALL_GIL_TIME.fetch_add((gil_time * 1000.0) as u64, AtomicOrdering::Relaxed);
        PYTHON_CALL_PREP_TIME.fetch_add((prep_time * 1000.0) as u64, AtomicOrdering::Relaxed);
        PYTHON_CALL_INFER_TIME.fetch_add((infer_time * 1000.0) as u64, AtomicOrdering::Relaxed);
        PYTHON_CALL_EXTRACT_TIME.fetch_add((extract_time * 1000.0) as u64, AtomicOrdering::Relaxed);
        PYTHON_CALL_TOTAL_TIME.fetch_add((total_time * 1000.0) as u64, AtomicOrdering::Relaxed);
        
        Ok(evals)
    });
    
    result
}

#[cfg(feature = "neural")]
fn battle_format() -> &'static str {
    static FORMAT: OnceCell<String> = OnceCell::new();
    FORMAT
        .get_or_init(|| std::env::var("POKEENGINE_FORMAT").unwrap_or_else(|_| "gen9ou".to_string()))
        .as_str()
}

#[cfg(feature = "neural")]
fn model_name() -> &'static str {
    static MODEL: OnceCell<String> = OnceCell::new();
    MODEL
        .get_or_init(|| std::env::var("POKEENGINE_MODEL").unwrap_or_else(|_| "Abra".to_string()))
        .as_str()
}

#[cfg(feature = "neural")]
fn checkpoint() -> Option<i32> {
    std::env::var("POKEENGINE_CHECKPOINT")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
}

#[cfg(feature = "neural")]
fn get_runner(py: Python<'_>) -> PyResult<Py<PyAny>> {
    static RUNNER: OnceCell<Py<PyAny>> = OnceCell::new();
    RUNNER
        .get_or_try_init(|| {
            let module = py.import("poke_engine")?;
            let runner_cls = module.getattr("NeuralInferenceRunner")?;

            let kwargs = PyDict::new(py);
            if let Some(ckpt) = checkpoint() {
                kwargs.set_item("checkpoint", ckpt)?;
            }
            let runner = if kwargs.is_empty() {
                runner_cls.call_method1("from_pretrained", (model_name(),))?
            } else {
                runner_cls.call_method("from_pretrained", (model_name(),), Some(kwargs))?
            };

            runner.call_method0("reset")?;
            Ok(runner.into())
        })
        .map(|obj| obj.clone_ref(py))
}

#[cfg(feature = "neural")]
fn state_converter(py: Python<'_>) -> PyResult<Py<PyAny>> {
    static CONVERTER: OnceCell<Py<PyAny>> = OnceCell::new();
    CONVERTER
        .get_or_try_init(|| {
            let module = py.import("poke_engine")?;
            let func = module.getattr("_state_from_pointer")?;
            Ok(func.into())
        })
        .map(|func| func.clone_ref(py))
}

#[cfg(feature = "neural")]
fn state_to_python(py: Python<'_>, state: &State) -> PyResult<Py<PyAny>> {
    let converter = state_converter(py)?;
    let ptr = state as *const State as usize;
    let obj = converter.as_ref(py).call1((ptr,))?;
    Ok(obj.into())
}

#[cfg(feature = "neural")]
fn state_pointers(states: &[&State]) -> Vec<usize> {
    states
        .iter()
        .map(|state| *state as *const State as usize)
        .collect()
}
