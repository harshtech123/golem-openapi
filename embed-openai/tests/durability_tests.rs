// embed-openai/tests/durability_tests.rs

use embed_openai::durability_helper::Durability;
use embed_openai::embed::error_code;
use embed_openai::embed::error;
use durability; // imported host API types

/// When the host `begin_durable_function` is unavailable, `new` should return
/// an `Err(error)` with `code == internal_error`.
#[tokio::test]
async fn new_maps_begin_error_to_internal_error() {
    let result = Durability::new(
        "embed",
        "generate",
        durability::DurableFunctionType::WriteRemote,
    )
    .await;

    // We expect an Err(error) because begin_durable_function isn't actually implemented
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.code, error_code::internal_error);
    assert!(
        err.message.contains("Failed to begin durable function"),
        "unexpected message: {}",
        err.message
    );
}

/// When the host `end_durable_function` is unavailable, `persist` should return
/// an `Err(error)` with `code == internal_error`.
#[tokio::test]
async fn persist_maps_end_error_to_internal_error() {
    // Construct a Durability with dummy begin_index (the real host functions will still fail)
    let dur = Durability {
        interface: "embed",
        function: "generate",
        function_type: durability::DurableFunctionType::WriteRemote,
        begin_index: 0,
    };

    // Use a simple input/result pair
    let input = "dummy_input";
    let result: Result<String, String> = Ok("dummy_output".to_string());

    let persist_res = dur.persist(input, &result).await;
    assert!(persist_res.is_err());
    let err = persist_res.unwrap_err();
    assert_eq!(err.code, error_code::internal_error);
    assert!(
        err.message.contains("Failed to end durable function"),
        "unexpected message: {}",
        err.message
    );
}
