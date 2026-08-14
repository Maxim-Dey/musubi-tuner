# python balance_repeats.py 209 279 186 222
# python balance_repeats.py 209 279 186 222 -k 30
# python balance_repeats.py 209 279 186 222 --weights 1 0.2 1 1

import argparse
import sys


def spread(counts, weights, repeats):
    # Нормированная экспозиция: shown / w. При равных весах — обычный перекос.
    shown = [n * r / w for n, w, r in zip(counts, weights, repeats)]
    return max(shown) / min(shown) - 1


def main():
    parser = argparse.ArgumentParser(
        description="Подбор num_repeats для балансировки датасетов",
    )
    parser.add_argument(
        "counts",
        nargs="+",
        type=int,
        help="Число изображений в каждом датасете",
    )
    parser.add_argument(
        "-k",
        "--max-k",
        type=int,
        default=20,
        help="Максимальный множитель k (по умолчанию 20)",
    )
    parser.add_argument(
        "-w",
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Желаемые веса датасетов (тот же порядок, что counts)",
    )
    args = parser.parse_args()

    if any(n <= 0 for n in args.counts):
        print("counts must be positive integers", file=sys.stderr)
        sys.exit(1)

    counts = args.counts
    if args.weights is None:
        weights = [1.0] * len(counts)
    else:
        weights = args.weights
        if len(weights) != len(counts):
            print(
                "weights must match counts length",
                file=sys.stderr,
            )
            sys.exit(1)
        if any(w <= 0 for w in weights):
            print("weights must be positive", file=sys.stderr)
            sys.exit(1)

    best = None
    for k in range(1, args.max_k + 1):
        target = k * max(n / w for n, w in zip(counts, weights))
        repeats = [
            max(1, round(target * w / n))
            for n, w in zip(counts, weights)
        ]
        score = spread(counts, weights, repeats)
        if best is None or score < spread(counts, weights, best):
            best = repeats
        shown = [n * r for n, r in zip(counts, repeats)]
        print(k, repeats, shown, f"{score:.1%}")

    print(
        "best:",
        best,
        [n * r for n, r in zip(counts, best)],
        f"{spread(counts, weights, best):.1%}",
    )


if __name__ == "__main__":
    main()
