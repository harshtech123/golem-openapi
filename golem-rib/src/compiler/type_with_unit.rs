// Copyright 2024-2025 Golem Cloud
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{GetTypeHint, InferredType};
use bincode::{Decode, Encode};
use golem_wasm_ast::analysis::analysed_type::{bool, field, option, record, tuple};
use golem_wasm_ast::analysis::{
    AnalysedResourceId, AnalysedResourceMode, AnalysedType, NameOptionTypePair, NameTypePair,
    TypeBool, TypeChr, TypeEnum, TypeF32, TypeF64, TypeFlags, TypeHandle, TypeList, TypeOption,
    TypeRecord, TypeResult, TypeS16, TypeS32, TypeS64, TypeS8, TypeStr, TypeTuple, TypeU16,
    TypeU32, TypeU64, TypeU8, TypeVariant,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum AnalysedTypeWithUnit {
    Unit,
    Type(AnalysedType),
}

impl AnalysedTypeWithUnit {
    pub fn unit() -> Self {
        AnalysedTypeWithUnit::Unit
    }

    pub fn analysed_type(typ: AnalysedType) -> Self {
        AnalysedTypeWithUnit::Type(typ)
    }
}

impl TryFrom<AnalysedTypeWithUnit> for AnalysedType {
    type Error = String;

    fn try_from(value: AnalysedTypeWithUnit) -> Result<Self, Self::Error> {
        match value {
            AnalysedTypeWithUnit::Unit => Ok(tuple(vec![])),
            AnalysedTypeWithUnit::Type(typ) => Ok(typ),
        }
    }
}

impl TryFrom<&InferredType> for AnalysedType {
    type Error = String;

    fn try_from(value: &InferredType) -> Result<Self, Self::Error> {
        let with_unit = AnalysedTypeWithUnit::try_from(value)?;
        AnalysedType::try_from(with_unit)
    }
}

impl TryFrom<&InferredType> for AnalysedTypeWithUnit {
    type Error = String;

    fn try_from(inferred_type: &InferredType) -> Result<Self, Self::Error> {
        match inferred_type {
            InferredType::Instance { .. } => {
                Err("Cannot convert Instance type to AnalysedType".to_string())
            }
            InferredType::Range { from, to } => {
                let from: AnalysedType = AnalysedType::try_from(from.as_ref())?;
                let to: Option<AnalysedType> = to
                    .as_ref()
                    .map(|t| AnalysedType::try_from(t.as_ref()))
                    .transpose()?;
                let analysed_type = match (from, to) {
                    (from_type, Some(to_type)) => record(vec![
                        field("from", option(from_type)),
                        field("to", option(to_type)),
                        field("inclusive", bool()),
                    ]),

                    (from_type, None) => record(vec![
                        field("from", option(from_type)),
                        field("inclusive", bool()),
                    ]),
                };
                Ok(AnalysedTypeWithUnit::analysed_type(analysed_type))
            }
            InferredType::Bool => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::Bool(
                TypeBool,
            ))),
            InferredType::S8 => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::S8(
                TypeS8,
            ))),
            InferredType::U8 => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::U8(
                TypeU8,
            ))),
            InferredType::S16 => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::S16(
                TypeS16,
            ))),
            InferredType::U16 => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::U16(
                TypeU16,
            ))),
            InferredType::S32 => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::S32(
                TypeS32,
            ))),
            InferredType::U32 => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::U32(
                TypeU32,
            ))),
            InferredType::S64 => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::S64(
                TypeS64,
            ))),
            InferredType::U64 => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::U64(
                TypeU64,
            ))),
            InferredType::F32 => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::F32(
                TypeF32,
            ))),
            InferredType::F64 => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::F64(
                TypeF64,
            ))),
            InferredType::Chr => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::Chr(
                TypeChr,
            ))),
            InferredType::Str => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::Str(
                TypeStr,
            ))),
            InferredType::List(inferred_type) => Ok(AnalysedTypeWithUnit::analysed_type(
                AnalysedType::List(TypeList {
                    inner: Box::new(inferred_type.as_ref().try_into()?),
                }),
            )),
            InferredType::Tuple(tuple) => Ok(AnalysedTypeWithUnit::analysed_type(
                AnalysedType::Tuple(TypeTuple {
                    items: tuple
                        .iter()
                        .map(|t| t.try_into())
                        .collect::<Result<Vec<AnalysedType>, String>>()?,
                }),
            )),
            InferredType::Record(record) => Ok(AnalysedTypeWithUnit::analysed_type(
                AnalysedType::Record(TypeRecord {
                    fields: record
                        .iter()
                        .map(|(name, typ)| {
                            Ok(NameTypePair {
                                name: name.to_string(),
                                typ: typ.try_into()?,
                            })
                        })
                        .collect::<Result<Vec<NameTypePair>, String>>()?,
                }),
            )),
            InferredType::Flags(flags) => Ok(AnalysedTypeWithUnit::analysed_type(
                AnalysedType::Flags(TypeFlags {
                    names: flags.clone(),
                }),
            )),
            InferredType::Enum(enums) => Ok(AnalysedTypeWithUnit::analysed_type(
                AnalysedType::Enum(TypeEnum {
                    cases: enums.clone(),
                }),
            )),
            InferredType::Option(option) => Ok(AnalysedTypeWithUnit::analysed_type(
                AnalysedType::Option(TypeOption {
                    inner: Box::new(option.as_ref().try_into()?),
                }),
            )),
            InferredType::Result { ok, error } => Ok(AnalysedTypeWithUnit::analysed_type(
                // In the case of result, there are instances users give just 1 value with zero function calls, we need to be flexible here
                AnalysedType::Result(TypeResult {
                    ok: ok
                        .as_ref()
                        .and_then(|t| t.as_ref().try_into().ok().map(Box::new)),
                    err: error
                        .as_ref()
                        .and_then(|t| t.as_ref().try_into().ok().map(Box::new)),
                }),
            )),
            InferredType::Variant(variant) => Ok(AnalysedTypeWithUnit::analysed_type(
                AnalysedType::Variant(TypeVariant {
                    cases: variant
                        .iter()
                        .map(|(name, typ)| {
                            Ok(NameOptionTypePair {
                                name: name.clone(),
                                typ: typ.as_ref().map(|t| t.try_into()).transpose()?,
                            })
                        })
                        .collect::<Result<Vec<NameOptionTypePair>, String>>()?,
                }),
            )),
            InferredType::Resource {
                resource_id,
                resource_mode,
            } => Ok(AnalysedTypeWithUnit::analysed_type(AnalysedType::Handle(
                TypeHandle {
                    resource_id: AnalysedResourceId(*resource_id),
                    mode: if resource_mode == &0 {
                        AnalysedResourceMode::Owned
                    } else {
                        AnalysedResourceMode::Borrowed
                    },
                },
            ))),

            InferredType::OneOf(_) => Err("ambiguous types".to_string()),
            InferredType::AllOf(types) => Err(format!(
                "ambiguous types {}",
                types
                    .iter()
                    .map(|x| x.get_type_hint().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            InferredType::Unknown => Err("failed to infer type".to_string()),
            // We don't expect to have a sequence type in the inferred type.as
            // This implies Rib will not support multiple types from worker-function results
            InferredType::Sequence(vec) => {
                if vec.is_empty() {
                    Ok(AnalysedTypeWithUnit::unit())
                } else if vec.len() == 1 {
                    let first = &vec[0];
                    Ok(first.try_into()?)
                } else {
                    Err("function with multiple return types not supported".to_string())
                }
            }
        }
    }
}
