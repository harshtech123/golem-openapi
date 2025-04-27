// src/openai.rs

use embed::{
    config, embedding_response, embedding, usage,
    rerank_response, rerank_result,
    error, error_code,
};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::env;

//
// Request/response shapes for the OpenAI API
//

#[derive(Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[derive(Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    model: String,
    data: Vec<OpenAIEmbedding>,
    usage: Option<OpenAIUsage>,
}

//
// Embeddings call
//

pub async fn call_embeddings(
    texts: &[String],
    cfg: &config,
) -> Result<embedding_response, error> {
    // Load the API key
    let api_key = env::var("OPENAI_API_KEY").map_err(|_| error {
        code: error_code::invalid_request,
        message: "OPENAI_API_KEY env var not set".into(),
        provider_error_json: None,
    })?;

    // Choose model (default to text-embedding-3-large)
    let model = cfg
        .model
        .as_ref()
        .map(String::as_str)
        .unwrap_or("text-embedding-3-large");

    // Build request payload
    let req_body = EmbeddingsRequest {
        model,
        input: texts,
        user: cfg.user.as_deref(),
    };

    // Send HTTP request
    let client = Client::new();
    // read base URL (for tests) or default
    let base = env::var("OPENAI_API_BASE_URL")
    .unwrap_or_else(|_| "https://api.openai.com".to_string());
    let url = format!("{}/v1/embeddings", base);
    let resp = client
    .post(&url)
    .bearer_auth(api_key)
    .json(&req_body)
    .send()
    .await
    .map_err(|e| error {
        code: error_code::provider_error,
        message: format!("HTTP request failed: {}", e),
        provider_error_json: None,
    })?;


    // Handle non-200
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(error {
            code: error_code::provider_error,
            message: format!("OpenAI API returned HTTP {}", status),
            provider_error_json: Some(body),
        });
    }

    // Parse JSON
    let body: OpenAIResponse = resp.json().await.map_err(|e| error {
        code: error_code::provider_error,
        message: format!("Failed to parse JSON: {}", e),
        provider_error_json: None,
    })?;

    // Map into WIT types
    let embeddings: Vec<embedding> = body
        .data
        .into_iter()
        .enumerate()
        .map(|(i, item)| embedding {
            index: i as u32,
            vector: item.embedding,
        })
        .collect();

    let usage_rec = body.usage.map(|u| usage {
        input_tokens: Some(u.prompt_tokens),
        total_tokens: Some(u.total_tokens),
    });

    Ok(embedding_response {
        embeddings,
        usage: usage_rec,
        model: body.model,
        provider_metadata_json: None,
    })
}

//
// Rerank emulation via cosine similarity
//

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norma = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let normb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norma == 0.0 || normb == 0.0 {
        0.0
    } else {
        dot / (norma * normb)
    }
}

pub async fn call_rerank(
    query: &String,
    documents: &[String],
    cfg: &config,
) -> Result<rerank_response, error> {
    // Combine query + documents into one embedding call
    let mut inputs = Vec::with_capacity(1 + documents.len());
    inputs.push(query.clone());
    inputs.extend_from_slice(documents);

    let resp = call_embeddings(&inputs, cfg).await?;

    if resp.embeddings.len() != inputs.len() {
        return Err(error {
            code: error_code::internal_error,
            message: "Embedding count mismatch".into(),
            provider_error_json: None,
        });
    }

    // Split off the query vector
    let mut iter = resp.embeddings.into_iter();
    let query_vec = iter.next().unwrap().vector;

    // Compute cosine similarity for each document
    let mut results = Vec::with_capacity(documents.len());
    for (i, emb) in iter.enumerate() {
        let score = cosine(&query_vec, &emb.vector);
        results.push(rerank_result {
            index: i as u32,
            relevance_score: score,
            document: Some(documents[i].clone()),
        });
    }

    Ok(rerank_response {
        results,
        usage: resp.usage,
        model: resp.model,
        provider_metadata_json: resp.provider_metadata_json,
    })
}
