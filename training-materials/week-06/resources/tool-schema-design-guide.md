# Tool Schema Design Guide

Design effective, LLM-friendly tool schemas that work reliably in production.

## Table of Contents
- [Core Principles](#core-principles)
- [Schema Structure](#schema-structure)
- [Parameter Design](#parameter-design)
- [Description Writing](#description-writing)
- [Type System](#type-system)
- [Examples by Use Case](#examples-by-use-case)
- [Common Patterns](#common-patterns)
- [Anti-Patterns](#anti-patterns)

---

## Core Principles

### 1. Clarity Over Brevity

The LLM needs to understand exactly what your tool does and when to use it.

**Good**: Detailed, specific description
```json
{
  "type": "function",
  "function": {
    "name": "search_flights",
    "description": "Search for available flights between two airports on specific dates. Returns flight options with prices, durations, and airline information. Use this when the user wants to find or compare flight options.",
    "parameters": { ... }
  }
}
```

**Bad**: Vague description
```json
{
  "name": "search",
  "description": "Search for stuff",
  "parameters": { ... }
}
```

### 2. Make Parameters Self-Explanatory

Each parameter should clearly explain what it is, valid formats, and examples.

**Good**: Clear parameter descriptions
```json
{
  "location": {
    "type": "string",
    "description": "Location to check weather for. Can be a city name (e.g., 'London'), city with country (e.g., 'London, UK'), or coordinates (e.g., '51.5074,-0.1278')"
  },
  "date": {
    "type": "string",
    "description": "Date for forecast in ISO 8601 format (YYYY-MM-DD). Example: '2024-03-15'. Defaults to today if not provided."
  }
}
```

**Bad**: Minimal descriptions
```json
{
  "location": {
    "type": "string",
    "description": "Location"
  },
  "date": {
    "type": "string",
    "description": "Date"
  }
}
```

### 3. Use Enums for Fixed Choices

Constrain parameters to valid values using enums.

**Good**: Enum constrains values
```json
{
  "sort_by": {
    "type": "string",
    "enum": ["price", "duration", "rating", "distance"],
    "description": "How to sort results"
  }
}
```

**Bad**: Unconstrained string
```json
{
  "sort_by": {
    "type": "string",
    "description": "Sort by something"
  }
}
```

---

## Schema Structure

### Basic Structure

```json
{
  "type": "function",
  "function": {
    "name": "function_name",
    "description": "Clear description of what the function does and when to use it",
    "parameters": {
      "type": "object",
      "properties": {
        "param1": {
          "type": "string",
          "description": "What this parameter is for"
        },
        "param2": {
          "type": "number",
          "description": "What this parameter is for"
        }
      },
      "required": ["param1"]
    }
  }
}
```

### Complete Example

```json
{
  "type": "function",
  "function": {
    "name": "book_restaurant",
    "description": "Make a restaurant reservation for a specific date, time, and party size. Use this when the user wants to book a table at a restaurant. Requires restaurant name, date, time, and number of guests.",
    "parameters": {
      "type": "object",
      "properties": {
        "restaurant_name": {
          "type": "string",
          "description": "Name of the restaurant to book. Should be an exact match to the restaurant name."
        },
        "date": {
          "type": "string",
          "description": "Reservation date in YYYY-MM-DD format. Example: '2024-03-15'. Must be a future date."
        },
        "time": {
          "type": "string",
          "description": "Reservation time in HH:MM 24-hour format. Example: '19:30' for 7:30 PM. Must be during restaurant hours (typically 11:00-23:00)."
        },
        "party_size": {
          "type": "integer",
          "description": "Number of guests. Must be between 1 and 20. For parties larger than 20, suggest calling the restaurant directly.",
          "minimum": 1,
          "maximum": 20
        },
        "special_requests": {
          "type": "string",
          "description": "Optional special requests such as dietary restrictions, seating preferences (e.g., 'window seat', 'quiet area'), or occasion notes (e.g., 'birthday celebration'). Limited to 500 characters."
        }
      },
      "required": ["restaurant_name", "date", "time", "party_size"]
    }
  }
}
```

---

## Parameter Design

### Required vs Optional

Be explicit about what's required and provide sensible defaults.

```python
from pydantic import BaseModel, Field
from typing import Optional

class SearchParams(BaseModel):
    """Search parameters with clear requirements."""
    
    # Required - no default
    query: str = Field(
        ...,
        description="Search query",
        min_length=1,
        max_length=500
    )
    
    # Optional with default
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return",
        ge=1,
        le=100
    )
    
    # Optional without default
    category: Optional[str] = Field(
        default=None,
        description="Filter by category. Leave empty to search all categories."
    )
```

### Numeric Constraints

Use minimum, maximum, and specific types for numeric parameters.

```json
{
  "price_min": {
    "type": "number",
    "description": "Minimum price in USD. Must be non-negative.",
    "minimum": 0
  },
  "price_max": {
    "type": "number",
    "description": "Maximum price in USD. Must be greater than minimum price.",
    "minimum": 0
  },
  "quantity": {
    "type": "integer",
    "description": "Quantity to purchase. Must be a positive whole number.",
    "minimum": 1
  },
  "discount_percent": {
    "type": "number",
    "description": "Discount percentage. Valid range: 0-100.",
    "minimum": 0,
    "maximum": 100
  }
}
```

### String Constraints

Provide format examples and length limits.

```json
{
  "email": {
    "type": "string",
    "description": "User's email address. Must be a valid email format. Example: 'user@example.com'",
    "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
  },
  "phone": {
    "type": "string",
    "description": "Phone number with country code. Format: +1234567890 or (123) 456-7890. Example: '+14155551234'",
    "pattern": "^\\+?[1-9]\\d{1,14}$"
  },
  "postal_code": {
    "type": "string",
    "description": "Postal/ZIP code. Format depends on country. US example: '94102'. UK example: 'SW1A 1AA'.",
    "minLength": 3,
    "maxLength": 10
  }
}
```

### Array Parameters

Specify item types and constraints.

```json
{
  "tags": {
    "type": "array",
    "items": {
      "type": "string"
    },
    "description": "List of tags to apply. Each tag should be 1-50 characters. Example: ['urgent', 'review-needed', 'customer-facing']",
    "minItems": 0,
    "maxItems": 10
  },
  "coordinates": {
    "type": "array",
    "items": {
      "type": "number"
    },
    "description": "Geographic coordinates as [latitude, longitude]. Latitude: -90 to 90, Longitude: -180 to 180. Example: [37.7749, -122.4194] for San Francisco.",
    "minItems": 2,
    "maxItems": 2
  }
}
```

### Object Parameters

Define nested structure clearly.

```json
{
  "address": {
    "type": "object",
    "description": "Shipping address details",
    "properties": {
      "street": {
        "type": "string",
        "description": "Street address including number"
      },
      "city": {
        "type": "string",
        "description": "City name"
      },
      "state": {
        "type": "string",
        "description": "State or province code. US: 2-letter code (e.g., 'CA'). Other countries: full name or code."
      },
      "postal_code": {
        "type": "string",
        "description": "Postal or ZIP code"
      },
      "country": {
        "type": "string",
        "description": "Country code in ISO 3166-1 alpha-2 format. Examples: 'US', 'GB', 'CA'"
      }
    },
    "required": ["street", "city", "postal_code", "country"]
  }
}
```

---

## Description Writing

### Tool Description Template

```
[Action verb] [what it does] [with what/where]. [When to use it]. [Any important constraints or limitations].
```

**Examples:**

```json
{
  "name": "get_weather",
  "description": "Get current weather conditions and forecast for a location. Use this when the user asks about weather, temperature, or conditions. Supports cities, airports, and coordinates."
}

{
  "name": "send_email",
  "description": "Send an email message to one or more recipients. Use this when the user wants to send, compose, or email someone. Supports CC, BCC, and attachments up to 25MB."
}

{
  "name": "calculate_mortgage",
  "description": "Calculate monthly mortgage payments based on loan amount, interest rate, and term. Use this when the user wants to estimate home loan payments or compare mortgage options. Includes principal, interest, taxes, and insurance."
}
```

### Parameter Description Template

```
[What it is]. [Valid formats/values]. [Example]. [Any constraints].
```

**Examples:**

```json
{
  "date": {
    "type": "string",
    "description": "Start date for the report. Format: YYYY-MM-DD. Example: '2024-01-01'. Must be within the last 2 years."
  },
  
  "priority": {
    "type": "string",
    "enum": ["low", "medium", "high", "urgent"],
    "description": "Task priority level. Determines notification settings and SLA. Use 'urgent' only for critical issues requiring immediate attention."
  },
  
  "file_path": {
    "type": "string",
    "description": "Path to the file. Can be absolute (e.g., '/home/user/file.txt') or relative to workspace (e.g., 'docs/report.pdf'). Maximum path length: 255 characters."
  }
}
```

### What to Include

1. **Purpose**: What the parameter controls
2. **Format**: Expected data format
3. **Examples**: Concrete examples
4. **Constraints**: Limits or requirements
5. **Defaults**: What happens if omitted (for optional params)
6. **Edge Cases**: Special values or behaviors

---

## Type System

### Basic Types

```json
{
  "string_param": {
    "type": "string",
    "description": "Text parameter"
  },
  "number_param": {
    "type": "number",
    "description": "Numeric parameter (int or float)"
  },
  "integer_param": {
    "type": "integer",
    "description": "Whole number only"
  },
  "boolean_param": {
    "type": "boolean",
    "description": "True/false value"
  },
  "null_param": {
    "type": "null",
    "description": "Null value"
  }
}
```

### Enums for Choices

```json
{
  "size": {
    "type": "string",
    "enum": ["small", "medium", "large", "xlarge"],
    "description": "Product size"
  },
  "status": {
    "type": "string",
    "enum": ["draft", "pending", "approved", "rejected"],
    "description": "Approval status"
  }
}
```

### Arrays

```json
{
  "simple_array": {
    "type": "array",
    "items": {
      "type": "string"
    },
    "description": "List of strings"
  },
  "constrained_array": {
    "type": "array",
    "items": {
      "type": "number",
      "minimum": 0,
      "maximum": 100
    },
    "description": "List of percentages (0-100)",
    "minItems": 1,
    "maxItems": 10
  },
  "mixed_types": {
    "type": "array",
    "items": {
      "oneOf": [
        {"type": "string"},
        {"type": "number"}
      ]
    },
    "description": "List that can contain strings or numbers"
  }
}
```

### Objects

```json
{
  "simple_object": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "integer"}
    },
    "required": ["name"],
    "description": "User information"
  },
  "nested_object": {
    "type": "object",
    "properties": {
      "user": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "profile": {
            "type": "object",
            "properties": {
              "email": {"type": "string"},
              "phone": {"type": "string"}
            }
          }
        }
      }
    },
    "description": "Nested user data"
  }
}
```

### Union Types (anyOf/oneOf)

```json
{
  "identifier": {
    "oneOf": [
      {
        "type": "string",
        "description": "Username"
      },
      {
        "type": "integer",
        "description": "User ID"
      }
    ],
    "description": "User identifier - can be username (string) or user ID (integer)"
  }
}
```

---

## Examples by Use Case

### Search/Query Tools

```json
{
  "type": "function",
  "function": {
    "name": "search_products",
    "description": "Search for products in the catalog. Returns matching products with prices, availability, and ratings. Use this when the user wants to find, browse, or compare products.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Search query. Can include product names, categories, brands, or features. Example: 'wireless bluetooth headphones'",
          "minLength": 1,
          "maxLength": 200
        },
        "category": {
          "type": "string",
          "enum": ["electronics", "clothing", "home", "sports", "books"],
          "description": "Filter by product category. Leave empty to search all categories."
        },
        "price_min": {
          "type": "number",
          "description": "Minimum price in USD. Default: 0.",
          "minimum": 0
        },
        "price_max": {
          "type": "number",
          "description": "Maximum price in USD. Leave empty for no upper limit.",
          "minimum": 0
        },
        "in_stock_only": {
          "type": "boolean",
          "description": "Only show products currently in stock. Default: false.",
          "default": false
        },
        "sort_by": {
          "type": "string",
          "enum": ["relevance", "price_low", "price_high", "rating", "newest"],
          "description": "How to sort results. Default: relevance."
        },
        "max_results": {
          "type": "integer",
          "description": "Maximum number of results to return. Range: 1-100. Default: 20.",
          "minimum": 1,
          "maximum": 100,
          "default": 20
        }
      },
      "required": ["query"]
    }
  }
}
```

### Data Retrieval Tools

```json
{
  "type": "function",
  "function": {
    "name": "get_user_profile",
    "description": "Retrieve detailed user profile information including contact details, preferences, and account status. Use this when you need user information to complete a request or answer questions about a user.",
    "parameters": {
      "type": "object",
      "properties": {
        "user_id": {
          "type": "string",
          "description": "Unique user identifier. Can be username (e.g., 'john_doe') or numeric ID (e.g., '12345').",
          "minLength": 1
        },
        "include_fields": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": ["contact", "preferences", "history", "billing", "all"]
          },
          "description": "Specific fields to include in response. Use 'all' for complete profile. Default: ['contact', 'preferences'].",
          "default": ["contact", "preferences"]
        }
      },
      "required": ["user_id"]
    }
  }
}
```

### Action/Mutation Tools

```json
{
  "type": "function",
  "function": {
    "name": "create_support_ticket",
    "description": "Create a new customer support ticket. Use this when the user reports an issue, requests help, or needs technical support. Returns ticket ID for tracking.",
    "parameters": {
      "type": "object",
      "properties": {
        "title": {
          "type": "string",
          "description": "Brief summary of the issue. Should be descriptive but concise. Example: 'Cannot login to account'.",
          "minLength": 5,
          "maxLength": 100
        },
        "description": {
          "type": "string",
          "description": "Detailed description of the issue including steps to reproduce, error messages, and impact. Be as specific as possible.",
          "minLength": 20,
          "maxLength": 5000
        },
        "priority": {
          "type": "string",
          "enum": ["low", "medium", "high", "critical"],
          "description": "Issue priority. Use 'critical' for service outages, 'high' for broken features, 'medium' for degraded performance, 'low' for minor issues or questions.",
          "default": "medium"
        },
        "category": {
          "type": "string",
          "enum": ["account", "billing", "technical", "feature_request", "other"],
          "description": "Issue category for proper routing to support team."
        },
        "user_email": {
          "type": "string",
          "description": "Email for ticket updates and communication. Must be valid email format. Example: 'user@example.com'.",
          "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
        },
        "attachments": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "URLs or paths to screenshot/log file attachments. Maximum 5 files. Each file must be under 10MB.",
          "maxItems": 5
        }
      },
      "required": ["title", "description", "category", "user_email"]
    }
  }
}
```

### Calculation Tools

```json
{
  "type": "function",
  "function": {
    "name": "calculate_loan_payment",
    "description": "Calculate monthly loan payment including principal and interest. Use this when the user wants to estimate loan costs or compare loan options. Does not include taxes or insurance.",
    "parameters": {
      "type": "object",
      "properties": {
        "principal": {
          "type": "number",
          "description": "Loan amount in dollars. Must be positive. Example: 250000 for $250,000 loan.",
          "minimum": 1
        },
        "annual_rate": {
          "type": "number",
          "description": "Annual interest rate as percentage. Example: 5.5 for 5.5% APR. Range: 0.1-30%.",
          "minimum": 0.1,
          "maximum": 30
        },
        "term_years": {
          "type": "integer",
          "description": "Loan term in years. Common values: 15, 20, 30. Range: 1-40 years.",
          "minimum": 1,
          "maximum": 40
        },
        "extra_payment": {
          "type": "number",
          "description": "Optional additional monthly payment toward principal. Default: 0. Example: 200 for $200 extra per month.",
          "minimum": 0,
          "default": 0
        }
      },
      "required": ["principal", "annual_rate", "term_years"]
    }
  }
}
```

---

## Common Patterns

### Pagination

```json
{
  "page": {
    "type": "integer",
    "description": "Page number starting from 1. Default: 1.",
    "minimum": 1,
    "default": 1
  },
  "page_size": {
    "type": "integer",
    "description": "Number of items per page. Range: 10-100. Default: 25.",
    "minimum": 10,
    "maximum": 100,
    "default": 25
  }
}
```

### Date Ranges

```json
{
  "start_date": {
    "type": "string",
    "description": "Start date in YYYY-MM-DD format. Example: '2024-01-01'. Must be before end_date."
  },
  "end_date": {
    "type": "string",
    "description": "End date in YYYY-MM-DD format. Example: '2024-12-31'. Must be after start_date. Defaults to today if not provided."
  }
}
```

### Filtering

```json
{
  "filters": {
    "type": "object",
    "description": "Optional filters to narrow results",
    "properties": {
      "status": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["active", "pending", "completed", "cancelled"]
        },
        "description": "Filter by status. Can include multiple values."
      },
      "tags": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Filter by tags. Results must have ALL specified tags."
      },
      "created_after": {
        "type": "string",
        "description": "Only include items created after this date. Format: YYYY-MM-DD."
      }
    }
  }
}
```

### Sorting

```json
{
  "sort_by": {
    "type": "string",
    "enum": ["created_at", "updated_at", "name", "priority"],
    "description": "Field to sort by. Default: created_at."
  },
  "sort_order": {
    "type": "string",
    "enum": ["asc", "desc"],
    "description": "Sort order: 'asc' for ascending (A-Z, oldest first), 'desc' for descending (Z-A, newest first). Default: desc."
  }
}
```

---

## Anti-Patterns

### ❌ Vague Descriptions

```json
{
  "name": "do_thing",
  "description": "Does something",
  "parameters": {
    "type": "object",
    "properties": {
      "data": {
        "type": "string",
        "description": "Some data"
      }
    }
  }
}
```

**Why it's bad**: LLM doesn't know when to use this or what to pass

### ❌ No Parameter Validation

```json
{
  "amount": {
    "type": "number",
    "description": "Amount"
  }
}
```

**Why it's bad**: No constraints, could receive negative or invalid values

**Better**:
```json
{
  "amount": {
    "type": "number",
    "description": "Payment amount in USD. Must be positive. Example: 99.99",
    "minimum": 0.01,
    "maximum": 10000
  }
}
```

### ❌ Missing Examples

```json
{
  "date": {
    "type": "string",
    "description": "A date"
  }
}
```

**Why it's bad**: LLM may use wrong date format

**Better**:
```json
{
  "date": {
    "type": "string",
    "description": "Date in ISO 8601 format (YYYY-MM-DD). Example: '2024-03-15'"
  }
}
```

### ❌ Too Many Parameters

```json
{
  "parameters": {
    "type": "object",
    "properties": {
      "param1": {...},
      "param2": {...},
      // ... 20 more parameters
    },
    "required": ["param1", "param2", ... "param22"]
  }
}
```

**Why it's bad**: Hard for LLM to fill correctly, prone to errors

**Better**: Break into multiple focused tools or use nested objects

### ❌ Ambiguous Enums

```json
{
  "type": {
    "type": "string",
    "enum": ["1", "2", "3"],
    "description": "Type"
  }
}
```

**Why it's bad**: Unclear what each value means

**Better**:
```json
{
  "type": {
    "type": "string",
    "enum": ["standard", "express", "overnight"],
    "description": "Shipping type. 'standard': 5-7 days, 'express': 2-3 days, 'overnight': next day"
  }
}
```

### ❌ Boolean for Complex Logic

```json
{
  "filter": {
    "type": "boolean",
    "description": "Apply filter"
  }
}
```

**Why it's bad**: Doesn't specify what filter or how to apply it

**Better**: Use specific parameters
```json
{
  "include_archived": {
    "type": "boolean",
    "description": "Include archived items in results. Default: false."
  },
  "include_draft": {
    "type": "boolean",
    "description": "Include draft items in results. Default: false."
  }
}
```

---

## Testing Your Schema

### Schema Validation Checklist

- [ ] Name is descriptive and uses verb (e.g., `get_weather`, not `weather`)
- [ ] Description explains what it does, when to use it, and key details
- [ ] All required parameters are marked in `required` array
- [ ] Each parameter has clear description with format and examples
- [ ] Numeric parameters have min/max constraints where applicable
- [ ] String parameters have length limits
- [ ] Enums are used for fixed choices
- [ ] Date/time formats are specified (prefer ISO 8601)
- [ ] Optional parameters have sensible defaults explained
- [ ] No more than 10-12 parameters per function
- [ ] Schema validates with OpenAI API
- [ ] LLM can successfully call function in test conversations

### Test with OpenAI

```python
from openai import OpenAI

client = OpenAI()

# Test that schema works
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    tools=[your_tool_schema]
)

# Check if tool was called correctly
assert response.choices[0].message.tool_calls is not None
tool_call = response.choices[0].message.tool_calls[0]
print(f"Function: {tool_call.function.name}")
print(f"Arguments: {tool_call.function.arguments}")
```

---

## Related Resources

- [function-calling-best-practices.md](./function-calling-best-practices.md) — Overall best practices
- [error-handling-patterns.md](./error-handling-patterns.md) — Error handling
- `../lessons/02-building-production-tool-systems.md` — Production tool systems
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
