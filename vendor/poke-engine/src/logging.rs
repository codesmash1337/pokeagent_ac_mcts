use std::sync::OnceLock;

const DEBUG_ENV_VAR: &str = "POKEENGINE_DEBUG_LOGS";

fn parse_bool(value: &str) -> bool {
    matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on" )
}

pub fn debug_logging_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(DEBUG_ENV_VAR)
            .map(|val| parse_bool(&val))
            .unwrap_or(false)
    })
}
