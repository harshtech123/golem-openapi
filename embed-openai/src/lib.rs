// 1) Bring in the generated `golem:embed` bindings
wit_bindgen_preview2::export!("wit/embed.wit");

// 2) Import the host durability interface
wit_bindgen_preview2::import!({
    path: "../wit/deps/golem-durability/golem-durability.wit",
    world: "durability"
});

// 3) Our modules
mod openai;
mod durability_helper;

use embed::*;                      // from the generated embed.wit
use durability::*;                 // host durability API (begin/end/persist)
use durability_helper::Durability; // our helper wrapper

/// Implements `embed::generate`
pub async fn generate(
    inputs: Vec<content_part>,
    cfg: config,
) -> Result<embedding_response, error> {
    let mut dur = Durability::new("embed", "generate", DurableFunctionType::WriteRemote)
        .await
        .map_err(|e| error {
            code: error_code::internal_error,
            message: format!("durability init failed: {}", e),
            provider_error_json: None,
        })?;

    let texts: Vec<String> = inputs
        .into_iter()
        .map(|p| match p {
            content_part::Text(t) => Ok(t),
            content_part::Image(_) => Err(error {
                code: error_code::unsupported,
                message: "image inputs not supported".into(),
                provider_error_json: None,
            }),
        })
        .collect::<Result<_, _>>()?;

    let result = openai::call_embeddings(&texts, &cfg).await;
    dur.persist(texts.clone(), &result).await.map_err(|e| error {
        code: error_code::internal_error,
        message: format!("durability persist failed: {}", e),
        provider_error_json: None,
    })?;

    result
}

/// Implements `embed::rerank`
pub async fn rerank(
    query: String,
    documents: Vec<String>,
    cfg: config,
) -> Result<rerank_response, error> {
    let mut dur = Durability::new("embed", "rerank", DurableFunctionType::WriteRemote)
        .await
        .map_err(|e| error {
            code: error_code::internal_error,
            message: format!("durability init failed: {}", e),
            provider_error_json: None,
        })?;

    let result = openai::call_rerank(&query, &documents, &cfg).await;
    // use the same `persist` as for generate
    dur.persist((query.clone(), documents.clone()), &result)
        .await
        .map_err(|e| error {
            code: error_code::internal_error,
            message: format!("durability persist failed: {}", e),
            provider_error_json: None,
        })?;

    result
}
