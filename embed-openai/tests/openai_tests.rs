// embed-openai/tests/openai_tests.rs

use httpmock::MockServer;
use httpmock::Method::POST;
use std::env;

// Import your functions under test.
// Assumes you have `pub mod openai;` in your `src/lib.rs`
use embed_openai::openai::{call_embeddings, call_rerank};
use embed_openai::embed::config;

#[tokio::test]
async fn call_embeddings_success() {
    // 1) Start a mock HTTP server
    let server = MockServer::start_async().await;

    // 2) Mock the embeddings endpoint
    let _m = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/v1/embeddings")
                .header("authorization", "Bearer test_key");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"
                {
                  "model": "text-embedding-3-large",
                  "data": [
                    { "embedding": [0.1, 0.2, 0.3] }
                  ],
                  "usage": { "prompt_tokens": 2, "total_tokens": 4 }
                }
                "#);
        })
        .await;

    // 3) Set environment variables so your code points at the mock
    env::set_var("OPENAI_API_KEY", "test_key");
    env::set_var("OPENAI_API_BASE_URL", &format!("http://{}", server.address()));

    // 4) Build a minimal config
    let cfg = config {
        model: None,
        task_type: None,
        dimensions: None,
        truncation: None,
        output_format: None,
        output_dtype: None,
        user: None,
        provider_options: vec![],
    };

    // 5) Call the function
    let texts = vec!["hello".to_string()];
    let resp = call_embeddings(&texts, &cfg).await.unwrap();

    // 6) Verify
    assert_eq!(resp.embeddings.len(), 1);
    assert_eq!(resp.embeddings[0].vector, vec![0.1, 0.2, 0.3]);
    let usage = resp.usage.unwrap();
    assert_eq!(usage.input_tokens.unwrap(), 2);
    assert_eq!(usage.total_tokens.unwrap(), 4);
}

#[tokio::test]
async fn call_embeddings_http_error() {
    let server = MockServer::start_async().await;
    let _m = server
        .mock_async(|when, then| {
            when.method(POST).path("/v1/embeddings");
            then.status(401).body("Unauthorized");
        })
        .await;

    env::set_var("OPENAI_API_KEY", "bad_key");
    env::set_var("OPENAI_API_BASE_URL", &format!("http://{}", server.address()));

    let cfg = config::default();
    let err = call_embeddings(&vec!["x".into()], &cfg).await.unwrap_err();

    // It should map to a provider_error
    assert_eq!(err.code, embed_openai::embed::error_code::provider_error);
    assert!(err.message.contains("HTTP 401"));
    assert_eq!(err.provider_error_json, Some("Unauthorized".to_string()));
}

#[tokio::test]
async fn call_rerank_cosine_similarity() {
    let server = MockServer::start_async().await;

    // Mock: two embeddings (query=[1,0], doc=[0,1])
    let _m = server
        .mock_async(|when, then| {
            when.method(POST).path("/v1/embeddings");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"
                {
                  "model": "text-embedding-3-large",
                  "data": [
                    { "embedding": [1.0, 0.0] },
                    { "embedding": [0.0, 1.0] }
                  ]
                }
                "#);
        })
        .await;

    env::set_var("OPENAI_API_KEY", "test_key");
    env::set_var("OPENAI_API_BASE_URL", &format!("http://{}", server.address()));

    let cfg = config::default();
    let query = "q".to_string();
    let docs = vec!["doc1".to_string()];
    let resp = call_rerank(&query, &docs, &cfg).await.unwrap();

    assert_eq!(resp.results.len(), 1);
    // cosine([1,0], [0,1]) == 0
    assert!((resp.results[0].relevance_score - 0.0).abs() < 1e-6);
    assert_eq!(resp.results[0].document, Some("doc1".to_string()));
}
