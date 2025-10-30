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
    pub value: f32,
    pub policy: Vec<f32>,
}

/// Returns a value estimate in `[0, 1]` for a given state. When the `neural` feature
/// is disabled, this function always returns `None` so callers can fall back to the
/// hand-crafted evaluation function.
pub fn neural_state_value(state: &State) -> Option<NeuralEvaluation> {
    #[cfg(feature = "neural")]
    {
        match python_state_value(state) {
            Ok(mut eval) => {
                eval.value = normalize_to_unit_interval(eval.value);
                Some(eval)
            }
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
        match python_state_value_with_perspective(state, side) {
            Ok(mut eval) => {
                eval.value = normalize_to_unit_interval(eval.value);
                Some(eval)
            }
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
                                Err(e) => {
                                    eprintln!("[TOKEN DECODE ERROR] Failed to get decode method: {}", e);
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[TOKEN DECODE ERROR] Failed to call get_tokenizer: {}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[TOKEN DECODE ERROR] Failed to get get_tokenizer function: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            eprintln!("[TOKEN DECODE ERROR] Failed to import poke_engine: {}", e);
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

#[cfg(feature = "neural")]
// pub fn normalize_to_unit_interval(raw: f32) -> f32 {
//     let s = 900.0;
//     let v = (raw / s).tanh();        // [-1, 1]
//     ((v + 1.0) * 0.5).clamp(1e-6, 1.0 - 1e-6)
// }
fn normalize_to_unit_interval(value: f32) -> f32 {
    const MIN: f32 = -1_100.0;
    const MAX: f32 = 1_100.0;
    ((value - MIN) / (MAX - MIN)).clamp(0.0, 1.0)
}

#[cfg(feature = "neural")]
fn python_state_value(state: &State) -> PyResult<NeuralEvaluation> {
    Python::with_gil(|py| {
        let runner = get_runner(py)?;
        let py_state = state_to_python(py, state)?;

        runner.as_ref(py).call_method0("reset")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("battle_format", battle_format())?;

        let result = runner
            .as_ref(py)
            .call_method("infer", (py_state.as_ref(py),), Some(kwargs))?;

        let state_value = result.getattr("state_value")?;
        let value: f32 = state_value.call_method0("item")?.extract()?;

        let policy_prior = result.getattr("policy_prior")?;
        let policy: Vec<f32> = policy_prior.extract()?;
        Ok(NeuralEvaluation { value, policy })
    })
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
            let mut value: f32 = state_value.call_method0("item")?.extract()?;
            value = normalize_to_unit_interval(value);
            let policy_prior = item.getattr("policy_prior")?;
            let policy: Vec<f32> = policy_prior.extract()?;
            evals.push(NeuralEvaluation { value, policy });
        }
        Ok(evals)
    })
}

#[cfg(feature = "neural")]
fn python_state_value_with_perspective(
    state: &State,
    side: SideReference,
) -> PyResult<NeuralEvaluation> {
    Python::with_gil(|py| {
        let runner = get_runner(py)?;
        let py_state = state_to_python(py, state)?;
        runner.as_ref(py).call_method0("reset")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("battle_format", battle_format())?;
        kwargs.set_item(
            "perspective",
            match side {
                SideReference::SideOne => "side_one",
                SideReference::SideTwo => "side_two",
            },
        )?;
        let result = runner
            .as_ref(py)
            .call_method("infer", (py_state.as_ref(py),), Some(kwargs))?;
        let state_value = result.getattr("state_value")?;
        let value: f32 = state_value.call_method0("item")?.extract()?;
        let policy_prior = result.getattr("policy_prior")?;
        let policy: Vec<f32> = policy_prior.extract()?;
        Ok(NeuralEvaluation { value, policy })
    })
}

#[cfg(feature = "neural")]
pub fn python_state_values_batch_with_perspective(
    states: &[&State],
    side: SideReference,
) -> PyResult<Vec<NeuralEvaluation>> {
    Python::with_gil(|py| {
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
            let mut value: f32 = state_value.call_method0("item")?.extract()?;
            value = normalize_to_unit_interval(value);
            let policy_prior = item.getattr("policy_prior")?;
            let policy: Vec<f32> = policy_prior.extract()?;
            evals.push(NeuralEvaluation { value, policy });
        }
        Ok(evals)
    })
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
