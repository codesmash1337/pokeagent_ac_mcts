//! Neural evaluation helpers that delegate to the Python `NeuralInferenceRunner`.

use crate::state::State;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};

static NEURAL_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static NEURAL_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static NEURAL_PY_CALLS: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "neural")]
use once_cell::sync::{Lazy, OnceCell};
#[cfg(feature = "neural")]
use std::collections::HashMap;
#[cfg(feature = "neural")]
use std::sync::Mutex;
#[cfg(feature = "neural")]
use pyo3::{prelude::*, types::PyDict};

#[cfg(feature = "neural")]
static STATE_VALUE_CACHE: Lazy<Mutex<HashMap<String, NeuralEvaluation>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

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
        if let Some(value) = {
            let cache = STATE_VALUE_CACHE.lock().expect("cache poisoned");
            cache.get(&serialized).cloned()
        } {
            NEURAL_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return Some(value);
        }

        NEURAL_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);

        match python_state_value(&serialized) {
            Ok(mut eval) => {
                eval.value = normalize_to_unit_interval(eval.value);
                if let Ok(mut cache) = STATE_VALUE_CACHE.lock() {
                    cache.insert(serialized, eval.clone());
                }
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

/// Returns stats about evaluator usage since the last call.
pub struct NeuralStats {
    pub py_calls: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

pub fn take_neural_stats() -> NeuralStats {
    NeuralStats {
        py_calls: NEURAL_PY_CALLS.swap(0, Ordering::Relaxed),
        cache_hits: NEURAL_CACHE_HITS.swap(0, Ordering::Relaxed),
        cache_misses: NEURAL_CACHE_MISSES.swap(0, Ordering::Relaxed),
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
        let runner = get_runner(py)?;
        let state_cls = get_state_class(py)?;

        // Reset the runner's internal RL2 state so each evaluation is independent.
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
            let runner = if kwargs.len() == 0 {
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
