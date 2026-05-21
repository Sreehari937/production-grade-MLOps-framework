from prefect import flow
from monitoring_flow import monitoring_flow
from pipeline_steps import (
    ingest_data,
    create_features,
    train_model,
    validate_model,
    register_model,
)

@flow(name="main-pipeline")
def main_flow():
    print("=== Starting main pipeline ===")

    # Step 5.2 — modular pipeline steps
    train, val, test               = ingest_data()
    X_tr, y_tr, X_v, y_v, X_te, y_te = create_features(train, val, test)
    model, run_id                  = train_model(X_tr, y_tr)
    passed, test_auc               = validate_model(model, run_id, X_v, y_v, X_te, y_te)
    register_model(run_id, passed, test_auc)

    # Step 5.3 — drift check, retrain if needed
    drift = monitoring_flow()
    if drift:
        print("Drift detected — rerunning training steps")
        train, val, test               = ingest_data()
        X_tr, y_tr, X_v, y_v, X_te, y_te = create_features(train, val, test)
        model, run_id                  = train_model(X_tr, y_tr)
        passed, test_auc               = validate_model(model, run_id, X_v, y_v, X_te, y_te)
        register_model(run_id, passed, test_auc)

    print("=== Pipeline complete ===")

if __name__ == "__main__":
    main_flow()