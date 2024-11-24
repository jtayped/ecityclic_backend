from flask import Flask
from routes.prediction import prediction_bp
from routes.tramit import tramit_bp

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(prediction_bp)
app.register_blueprint(tramit_bp)

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
