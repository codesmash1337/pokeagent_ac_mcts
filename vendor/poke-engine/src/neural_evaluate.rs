//! Neural evaluation helpers that delegate to the Python `NeuralInferenceRunner`.

use crate::state::State;

#[cfg(feature = "neural")]
use once_cell::sync::OnceCell;
#[cfg(feature = "neural")]
use pyo3::{prelude::*, types::PyDict};
#[cfg(feature = "neural")]
use std::time::Instant;

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
                eprintln!("neural evaluation failed: {message}");
                None
            }
        }
    }
    #[cfg(not(feature = "neural"))]
    {
        let _ = state; // silence warnings
        eprintln!("neural feature disabled: enable \"neural\" feature to use the critic evaluator");
        None
    }
}

#[cfg(feature = "neural")]
fn normalize_to_unit_interval(value: f32) -> f32 {
    const MIN: f32 = -1_100.0;
    const MAX: f32 = 1_100.0;
    ((value - MIN) / (MAX - MIN)).clamp(0.0, 1.0)
}

#[cfg(feature = "neural")]
fn python_state_value(serialized: &str) -> PyResult<NeuralEvaluation> {
    Python::with_gil(|py| {
        let total_start = Instant::now();
        eprintln!("[neural_evaluate] acquiring runner");
        let runner = get_runner(py)?;
        let state_cls = get_state_class(py)?;

        eprintln!("[neural_evaluate] resetting runner");
        let reset_start = Instant::now();
        // Reset the runner's internal RL2 state so each evaluation is independent.
        // runner.as_ref(py).call_method0("reset")?;
        eprintln!(
            "[neural_evaluate] reset took {:.3}ms",
            reset_start.elapsed().as_secs_f64() * 1_000.0
        );

        let py_state = state_cls
            .as_ref(py)
            .call_method1("from_string", (serialized,))?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("battle_format", battle_format())?;

        eprintln!("[neural_evaluate] running infer");
        let infer_start = Instant::now();
        let result = runner
            .as_ref(py)
            .call_method("infer", (py_state,), Some(kwargs))?;
        eprintln!(
            "[neural_evaluate] infer took {:.3}ms",
            infer_start.elapsed().as_secs_f64() * 1_000.0
        );

        let state_value = result.getattr("state_value")?;
        let value: f32 = state_value.call_method0("item")?.extract()?;

        let policy_prior = result.getattr("policy_prior")?;
        let policy: Vec<f32> = policy_prior.extract()?;

        eprintln!(
            "[neural_evaluate] total eval {:.3}ms",
            total_start.elapsed().as_secs_f64() * 1_000.0
        );
        Ok(NeuralEvaluation { value, policy })
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
