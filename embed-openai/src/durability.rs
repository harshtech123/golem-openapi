// src/durability.rs

use durability;                         // the imported golem-durability module
use embed::{error, error_code};        // your WIT types
use golem_wasm_rpc::IntoValueAndType;   // for typed persistence

/// A helper to wrap each embedding/rerank call in a durable‐function.
pub struct Durability {
    interface: &'static str,
    function:  &'static str,
    function_type: durability::DurableFunctionType,
    begin_index:    u32,
}

impl Durability {
    /// Start a new durable‐function invocation.
    pub async fn new(
        interface: &'static str,
        function:  &'static str,
        function_type: durability::DurableFunctionType,
    ) -> Result<Self, error> {
        // Record the call (metrics/logs)
        let _ = durability::observe_function_call(interface.to_string(), function.to_string())
            .await;

        // Begin the durable scope
        let idx = durability::begin_durable_function(function_type)
            .await
            .map_err(|_| error {
                code: error_code::internal_error,
                message: "Failed to begin durable function".into(),
                provider_error_json: None,
            })?;

        Ok(Self { interface, function, function_type, begin_index: idx })
    }

    /// Persist the input + result pair, then end the durable‐function.
    ///
    /// `input` and the `Ok` or `Err` value must implement `IntoValueAndType`.
    pub async fn persist<I, O, E>(
        &self,
        input: I,
        result: &Result<O, E>,
    ) -> Result<(), error>
    where
        I: IntoValueAndType,
        O: IntoValueAndType + Clone,
        E: IntoValueAndType + Clone,
    {
        // Fully‐qualified function name for logs (e.g. "embed::generate")
        let fn_name = if self.interface.is_empty() {
            self.function.to_string()
        } else {
            format!("{}::{}", self.interface, self.function)
        };

        // Convert to host‐serializable `ValueAndType`
        let req_vt = input.into_value_and_type();
        let resp_vt = match result {
            Ok(o) => o.clone().into_value_and_type(),
            Err(e) => e.clone().into_value_and_type(),
        };

        // Persist the invocation (typed)
        let _ = durability::persist_typed_durable_function_invocation(
            fn_name.clone(),
            req_vt,
            resp_vt,
            self.function_type,
        )
        .await;

        // End the durable function (no forced commit)
        durability::end_durable_function(self.function_type, self.begin_index, false)
            .await
            .map_err(|_| error {
                code: error_code::internal_error,
                message: "Failed to end durable function".into(),
                provider_error_json: None,
            })?;

        Ok(())
    }
}
