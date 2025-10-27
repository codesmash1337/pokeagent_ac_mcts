//! Neural evaluation helpers that delegate to the Python `NeuralInferenceRunner`.

use crate::state::{SideReference, State};

#[cfg(feature = "neural")]
use once_cell::sync::OnceCell;
#[cfg(feature = "neural")]
use pyo3::{prelude::*, types::PyDict, types::PyList};
const VERBOSE_NEURAL_EVAL: bool = false;

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
        let serialized = state.serialize();
        match python_state_value(&serialized) {
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
        let serialized: Vec<String> = states.iter().map(|state| state.serialize()).collect();
        match python_state_values_batch(&serialized) {
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
        let serialized = state.serialize();
        match python_state_value_with_perspective(&serialized, side) {
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
        let serialized: Vec<String> = states.iter().map(|state| state.serialize()).collect();
        match python_state_values_batch_with_perspective(&serialized, side) {
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
fn python_state_value(serialized: &str) -> PyResult<NeuralEvaluation> {
    Python::with_gil(|py| {
        let runner = get_runner(py)?;

        let state_cls = get_state_class(py)?;

        runner.as_ref(py).call_method0("reset")?;

        let py_state = state_cls
            .as_ref(py)
            .call_method1("from_string", (serialized,))?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("battle_format", battle_format())?;

        let result = runner
            .as_ref(py)
            .call_method("infer", (py_state,), Some(kwargs))?;

        let state_value = result.getattr("state_value")?;
        let value: f32 = state_value.call_method0("item")?.extract()?;

        let policy_prior = result.getattr("policy_prior")?;
        let policy: Vec<f32> = policy_prior.extract()?;
        Ok(NeuralEvaluation { value, policy })
    })
}

#[cfg(feature = "neural")]
pub fn python_state_values_batch(serialized: &[String]) -> PyResult<Vec<NeuralEvaluation>> {
    Python::with_gil(|py| {
        let runner = get_runner(py)?;
        runner.as_ref(py).call_method0("reset")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("battle_format", battle_format())?;
        let py_states = PyList::new(py, serialized);
        let results = runner
            .as_ref(py)
            .call_method("infer_batch", (py_states,), Some(kwargs))?;
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
    serialized: &str,
    side: SideReference,
) -> PyResult<NeuralEvaluation> {
    Python::with_gil(|py| {
        let runner = get_runner(py)?;
        let state_cls = get_state_class(py)?;
        runner.as_ref(py).call_method0("reset")?;
        let py_state = state_cls
            .as_ref(py)
            .call_method1("from_string", (serialized,))?;
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
            .call_method("infer", (py_state,), Some(kwargs))?;
        let state_value = result.getattr("state_value")?;
        let value: f32 = state_value.call_method0("item")?.extract()?;
        let policy_prior = result.getattr("policy_prior")?;
        let policy: Vec<f32> = policy_prior.extract()?;
        Ok(NeuralEvaluation { value, policy })
    })
}

#[cfg(feature = "neural")]
pub fn python_state_values_batch_with_perspective(
    serialized: &[String],
    side: SideReference,
) -> PyResult<Vec<NeuralEvaluation>> {
    Python::with_gil(|py| {
        let runner = get_runner(py)?;
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
        let py_states = PyList::new(py, serialized);
        let results = runner
            .as_ref(py)
            .call_method("infer_batch", (py_states,), Some(kwargs))?;
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

            let mut kwargs = PyDict::new(py);
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
fn get_state_class(py: Python<'_>) -> PyResult<Py<PyAny>> {
    static STATE_CLS: OnceCell<Py<PyAny>> = OnceCell::new();
    STATE_CLS
        .get_or_try_init(|| {
            let module = py.import("poke_engine")?;
            let state_cls = module.getattr("State")?;
            Ok(state_cls.into())
        })
        .map(|cls| cls.clone_ref(py))
}
