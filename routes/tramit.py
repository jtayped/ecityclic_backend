from flask import Blueprint, request, jsonify
from model.utils import tramits_data, get_tramit_by_id
from urllib.parse import unquote
import unicodedata

tramit_bp = Blueprint("tramit", __name__)


@tramit_bp.route("/tramit/<tramit_id>")
def get_tramit(tramit_id):
    # Decode the ID to handle any URL-encoded characters
    decoded_id = unquote(tramit_id)

    print(decoded_id)

    tramit = get_tramit_by_id(decoded_id)
    if tramit:
        return jsonify(tramit)
    return jsonify({"error": "Data not found"}), 404


@tramit_bp.route("/search", methods=["GET"])
def search_titles():
    query = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 10))

    if not query:
        return {"error": "Query parameter 'q' is required"}, 400

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
    results = matches[["id", "title", "current"]].head(limit).to_dict(orient="records")

    if not matches.empty:
        return jsonify({"results": results, "total": len(results)})

    return jsonify({"message": "No matching titles found"}), 404
