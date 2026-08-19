"""assign a new customer record to its cluster using the saved model."""
import argparse
import joblib
import pandas as pd


def assign(model_path, age, income, spending):
    bundle = joblib.load(model_path)
    km = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle.get("features", ["Age", "Annual Income (k$)", "Spending Score (1-100)"])
    # build a one-row frame with the same column names the scaler was fit on
    # so we don't trip sklearn's feature-name warning
    x = pd.DataFrame([[age, income, spending]], columns=features)
    if scaler is not None:
        x = scaler.transform(x)
    else:
        x = x.values
    return int(km.predict(x)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/kmeans.joblib")
    ap.add_argument("--age", type=int, required=True)
    ap.add_argument("--income", type=float, required=True)
    ap.add_argument("--spending", type=float, required=True)
    args = ap.parse_args()
    c = assign(args.model, args.age, args.income, args.spending)
    print("cluster:", c)


if __name__ == "__main__":
    main()
