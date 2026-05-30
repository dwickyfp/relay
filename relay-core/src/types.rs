//! Type system for Relay.

use std::fmt;

/// Core type enumeration for Relay's internal type system.
/// Maps directly to Apache Arrow data types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelayType {
    // Primitive types
    Boolean,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,

    // String types
    Utf8,
    LargeUtf8,
    Utf8View,

    // Binary types
    Binary,
    LargeBinary,
    BinaryView,

    // Temporal types
    Date32,
    Date64,
    Time32Second,
    Time32Millisecond,
    Time64Microsecond,
    Time64Nanosecond,
    Timestamp(TimeUnit, Option<String>),
    Duration(TimeUnit),
    Interval(IntervalUnit),

    // Complex types
    List(Box<RelayType>),
    LargeList(Box<RelayType>),
    FixedSizeList(Box<RelayType>, i32),
    Struct(Vec<(String, RelayType)>),
    Map(Box<RelayType>, bool),
    Dictionary(Box<RelayType>, Box<RelayType>),
    Union(Vec<RelayType>),

    // Special
    Null,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntervalUnit {
    YearMonth,
    DayTime,
    MonthDayNano,
}

impl RelayType {
    /// Returns true if this type is a fixed-width primitive type.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            RelayType::Boolean
                | RelayType::Int8
                | RelayType::Int16
                | RelayType::Int32
                | RelayType::Int64
                | RelayType::UInt8
                | RelayType::UInt16
                | RelayType::UInt32
                | RelayType::UInt64
                | RelayType::Float16
                | RelayType::Float32
                | RelayType::Float64
                | RelayType::Date32
                | RelayType::Date64
        )
    }

    /// Returns true if this type is a variable-length string type.
    pub fn is_string(&self) -> bool {
        matches!(
            self,
            RelayType::Utf8 | RelayType::LargeUtf8 | RelayType::Utf8View
        )
    }

    /// Returns true if this type is a variable-length binary type.
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            RelayType::Binary | RelayType::LargeBinary | RelayType::BinaryView
        )
    }

    /// Returns true if this type is a temporal type.
    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            RelayType::Date32
                | RelayType::Date64
                | RelayType::Time32Second
                | RelayType::Time32Millisecond
                | RelayType::Time64Microsecond
                | RelayType::Time64Nanosecond
                | RelayType::Timestamp(_, _)
                | RelayType::Duration(_)
                | RelayType::Interval(_)
        )
    }

    /// Returns true if this type supports SIMD vectorized operations.
    pub fn is_simd_compatible(&self) -> bool {
        matches!(
            self,
            RelayType::Int8
                | RelayType::Int16
                | RelayType::Int32
                | RelayType::Int64
                | RelayType::UInt8
                | RelayType::UInt16
                | RelayType::UInt32
                | RelayType::UInt64
                | RelayType::Float16
                | RelayType::Float32
                | RelayType::Float64
        )
    }

    /// Returns the byte size of fixed-width types, or None for variable-width.
    pub fn byte_size(&self) -> Option<usize> {
        match self {
            RelayType::Boolean => Some(1), // bit-packed
            RelayType::Int8 | RelayType::UInt8 => Some(1),
            RelayType::Int16 | RelayType::UInt16 | RelayType::Float16 => Some(2),
            RelayType::Int32
            | RelayType::UInt32
            | RelayType::Float32
            | RelayType::Date32
            | RelayType::Time32Second
            | RelayType::Time32Millisecond => Some(4),
            RelayType::Int64
            | RelayType::UInt64
            | RelayType::Float64
            | RelayType::Date64
            | RelayType::Time64Microsecond
            | RelayType::Time64Nanosecond => Some(8),
            RelayType::Timestamp(_, _) | RelayType::Duration(_) => Some(8),
            _ => None,
        }
    }
}

impl fmt::Display for RelayType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RelayType::Boolean => write!(f, "bool"),
            RelayType::Int8 => write!(f, "i8"),
            RelayType::Int16 => write!(f, "i16"),
            RelayType::Int32 => write!(f, "i32"),
            RelayType::Int64 => write!(f, "i64"),
            RelayType::UInt8 => write!(f, "u8"),
            RelayType::UInt16 => write!(f, "u16"),
            RelayType::UInt32 => write!(f, "u32"),
            RelayType::UInt64 => write!(f, "u64"),
            RelayType::Float16 => write!(f, "f16"),
            RelayType::Float32 => write!(f, "f32"),
            RelayType::Float64 => write!(f, "f64"),
            RelayType::Utf8 => write!(f, "string"),
            RelayType::LargeUtf8 => write!(f, "large_string"),
            RelayType::Utf8View => write!(f, "string_view"),
            RelayType::Binary => write!(f, "binary"),
            RelayType::LargeBinary => write!(f, "large_binary"),
            RelayType::BinaryView => write!(f, "binary_view"),
            RelayType::Date32 => write!(f, "date32"),
            RelayType::Date64 => write!(f, "date64"),
            RelayType::Timestamp(u, tz) => {
                write!(f, "timestamp({}", u)?;
                if let Some(tz) = tz {
                    write!(f, ", \"{}\"", tz)?;
                }
                write!(f, ")")
            }
            RelayType::Duration(u) => write!(f, "duration({})", u),
            RelayType::List(inner) => write!(f, "list[{}]", inner),
            RelayType::LargeList(inner) => write!(f, "large_list[{}]", inner),
            RelayType::FixedSizeList(inner, size) => write!(f, "fixed_list[{}, {}]", inner, size),
            RelayType::Struct(fields) => {
                write!(f, "struct<")?;
                for (i, (name, ty)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", name, ty)?;
                }
                write!(f, ">")
            }
            RelayType::Null => write!(f, "null"),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TimeUnit::Second => write!(f, "s"),
            TimeUnit::Millisecond => write!(f, "ms"),
            TimeUnit::Microsecond => write!(f, "us"),
            TimeUnit::Nanosecond => write!(f, "ns"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_types() {
        assert!(RelayType::Int32.is_primitive());
        assert!(RelayType::Float64.is_primitive());
        assert!(RelayType::Boolean.is_primitive());
        assert!(!RelayType::Utf8.is_primitive());
        assert!(!RelayType::List(Box::new(RelayType::Int32)).is_primitive());
    }

    #[test]
    fn test_string_types() {
        assert!(RelayType::Utf8.is_string());
        assert!(RelayType::LargeUtf8.is_string());
        assert!(RelayType::Utf8View.is_string());
        assert!(!RelayType::Binary.is_string());
        assert!(!RelayType::Int32.is_string());
    }

    #[test]
    fn test_binary_types() {
        assert!(RelayType::Binary.is_binary());
        assert!(RelayType::LargeBinary.is_binary());
        assert!(RelayType::BinaryView.is_binary());
        assert!(!RelayType::Utf8.is_binary());
    }

    #[test]
    fn test_temporal_types() {
        assert!(RelayType::Date32.is_temporal());
        assert!(RelayType::Timestamp(TimeUnit::Nanosecond, None).is_temporal());
        assert!(RelayType::Duration(TimeUnit::Millisecond).is_temporal());
        assert!(!RelayType::Int64.is_temporal());
    }

    #[test]
    fn test_simd_compatible() {
        assert!(RelayType::Float64.is_simd_compatible());
        assert!(RelayType::Int32.is_simd_compatible());
        assert!(!RelayType::Utf8.is_simd_compatible());
        assert!(!RelayType::Boolean.is_simd_compatible());
    }

    #[test]
    fn test_byte_size() {
        assert_eq!(RelayType::Int8.byte_size(), Some(1));
        assert_eq!(RelayType::Int16.byte_size(), Some(2));
        assert_eq!(RelayType::Int32.byte_size(), Some(4));
        assert_eq!(RelayType::Int64.byte_size(), Some(8));
        assert_eq!(RelayType::Float32.byte_size(), Some(4));
        assert_eq!(RelayType::Float64.byte_size(), Some(8));
        assert_eq!(RelayType::Utf8.byte_size(), None);
        assert_eq!(RelayType::Boolean.byte_size(), Some(1));
    }

    #[test]
    fn test_display_primitives() {
        assert_eq!(format!("{}", RelayType::Boolean), "bool");
        assert_eq!(format!("{}", RelayType::Int32), "i32");
        assert_eq!(format!("{}", RelayType::Float64), "f64");
        assert_eq!(format!("{}", RelayType::Utf8), "string");
    }

    #[test]
    fn test_display_temporal() {
        assert_eq!(format!("{}", RelayType::Date32), "date32");
        assert_eq!(
            format!("{}", RelayType::Timestamp(TimeUnit::Nanosecond, None)),
            "timestamp(ns)"
        );
        assert_eq!(
            format!(
                "{}",
                RelayType::Timestamp(TimeUnit::Millisecond, Some("UTC".to_string()))
            ),
            "timestamp(ms, \"UTC\")"
        );
    }

    #[test]
    fn test_display_complex() {
        assert_eq!(
            format!("{}", RelayType::List(Box::new(RelayType::Int32))),
            "list[i32]"
        );
        assert_eq!(
            format!(
                "{}",
                RelayType::FixedSizeList(Box::new(RelayType::Float64), 3)
            ),
            "fixed_list[f64, 3]"
        );
        assert_eq!(
            format!(
                "{}",
                RelayType::Struct(vec![
                    ("name".to_string(), RelayType::Utf8),
                    ("age".to_string(), RelayType::Int32),
                ])
            ),
            "struct<name: string, age: i32>"
        );
    }

    #[test]
    fn test_display_time_unit() {
        assert_eq!(format!("{}", TimeUnit::Second), "s");
        assert_eq!(format!("{}", TimeUnit::Millisecond), "ms");
        assert_eq!(format!("{}", TimeUnit::Microsecond), "us");
        assert_eq!(format!("{}", TimeUnit::Nanosecond), "ns");
    }
}
