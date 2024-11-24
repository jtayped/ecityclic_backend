from flask import Blueprint, request, jsonify
from model.utils import tramits_data, get_tramit_by_id
from urllib.parse import unquote
import unicodedata
import math

tramit_bp = Blueprint("tramit", __name__)


@tramit_bp.route("/tramit/<tramit_id>")
def get_tramit(tramit_id):
    # Decode the ID to handle any URL-encoded characters
    decoded_id = unquote(tramit_id)

    tramit = get_tramit_by_id(decoded_id)
    if tramit:
        return jsonify(tramit)
    return jsonify({"error": "Data not found"}), 404


@tramit_bp.route("/search", methods=["GET"])
def search_titles():
    # Get search parameters
    query = request.args.get("q", "").strip().lower()
    page = max(1, int(request.args.get("page", 1)))  # Ensure page is at least 1
    per_page = min(
        50, int(request.args.get("per_page", 10))
    )  # Limit maximum items per page

    # If query is provided, search for matches
    if query:
        # Normalize the query to remove accents
        normalized_query = unicodedata.normalize("NFD", query)
        normalized_query = "".join(
            char for char in normalized_query if unicodedata.category(char) != "Mn"
        )

        # Function to normalize titles in the dataset
        def normalize_title(title):
            normalized = unicodedata.normalize("NFD", title)
            return "".join(
                char for char in normalized if unicodedata.category(char) != "Mn"
            )

        # Normalize titles and search
        tramits_data["normalized_title"] = tramits_data["title"].apply(normalize_title)
        matches = tramits_data[
            tramits_data["normalized_title"].str.contains(
                normalized_query, case=False, na=False
            )
        ]
    else:
        # If no query, use all records
        matches = tramits_data

    # Calculate pagination metadata
    total_results = len(matches)
    total_pages = math.ceil(total_results / per_page)

    # Adjust page number if it exceeds total pages
    page = min(page, total_pages) if total_pages > 0 else 1

    # Calculate slice indices for pagination
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    # Get paginated results
    paginated_results = (
        matches[["id", "title", "current"]]
        .iloc[start_idx:end_idx]
        .to_dict(orient="records")
    )

    return jsonify(
        {
            "results": paginated_results,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_results": total_results,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            },
        }
    )
