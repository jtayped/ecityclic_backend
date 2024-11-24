# Tramit Prediction API Documentation

## Overview
This API provides endpoints for searching tramits (procedures/transactions), retrieving tramit details, and predicting the next likely tramit in a sequence using a trained LSTM model.

## Endpoints

### Search Tramits
Search for tramits by title.

```
GET /search
```

#### Query Parameters
| Parameter | Type    | Required | Description                                     |
|-----------|---------|----------|-------------------------------------------------|
| q         | string  | Yes      | Search query for tramit titles                  |
| limit     | integer | No       | Maximum number of results (default: 10)         |

#### Response Format
```json
{
    "results": [
        {
            "id": "fiYfkvA4rz/ML+AOCJ4FcFcF600sMiHFaHqvKV0hZuU=",
            "title": "Accions per a la defensa dels drets humans...",
            "current": false
        }
    ],
    "total": 1
}
```

#### Error Responses
- `400 Bad Request`: Missing query parameter
- `404 Not Found`: No matching titles found

#### Example Request
```bash
curl -X GET "http://localhost:5000/search?q=drets&limit=5"
```

### Get Tramit Details
Retrieve details for a specific tramit by ID.

```
GET /tramit/{tramit_id}
```

#### Path Parameters
| Parameter | Type   | Description                    |
|-----------|--------|--------------------------------|
| tramit_id | string | Base64-encoded tramit ID       |

#### Response Format
```json
{
    "id": "fiYfkvA4rz/ML+AOCJ4FcFcF600sMiHFaHqvKV0hZuU=",
    "title": "Accions per a la defensa dels drets humans...",
    "current": false
}
```

#### Error Responses
- `404 Not Found`: Tramit ID not found

#### Example Request
```bash
curl -X GET "http://localhost:5000/tramit/fiYfkvA4rz%2FML%2BAOCJ4FcFcF600sMiHFaHqvKV0hZuU%3D"
```

### Predict Next Tramit
Predict the next most likely tramit based on a sequence of previous tramits.

```
POST /predict
```

#### Request Body
```json
{
    "sequence": [
        "fiYfkvA4rz/ML+AOCJ4FcFcF600sMiHFaHqvKV0hZuU=",
        "another-tramit-id-here..."
    ]
}
```

#### Request Parameters
| Parameter | Type     | Required | Description                               |
|-----------|----------|----------|-------------------------------------------|
| sequence  | string[] | Yes      | Array of tramit IDs in sequential order   |

#### Response Format
```json
{
    "predicted_tramit": {
        "id": "predicted-tramit-id",
        "title": "Predicted Tramit Title",
        "current": true
    },
    "probability": 0.85
}
```

#### Error Responses
- `400 Bad Request`: Invalid input or missing sequence
- `400 Bad Request`: Unknown tramit ID in sequence

#### Example Request
```bash
curl -X POST "http://localhost:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "sequence": ["fiYfkvA4rz/ML+AOCJ4FcFcF600sMiHFaHqvKV0hZuU="]
         }'
```

## Technical Details

### Model Architecture
The API uses a trained LSTM model with the following specifications:
- Embedding dimension: 32
- Hidden dimension: 64
- Context size: 3
- 2-layer LSTM with dropout (0.2)
- Fully connected layers with ReLU activation

### Input Processing
- The model uses a context window of the last 3 tramits
- Sequences shorter than 3 are padded with zeros
- Longer sequences are truncated to the last 3 items
- Input tramit IDs are mapped to numerical indices using a pre-trained mapping

### Prediction Features
- Excludes previously seen tramits from predictions
- Renormalizes probabilities after masking
- Returns both the predicted tramit and its probability

## Notes
1. All tramit IDs are Base64-encoded and may contain special characters that need URL encoding in requests
2. The search function is accent-insensitive and case-insensitive
3. The prediction model uses a fixed context size of 3 previous tramits
4. The API returns JSON responses for all endpoints
5. Error responses include descriptive messages to help with debugging
