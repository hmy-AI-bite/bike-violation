"""读 ultralytics 的 results.csv，判断过拟合 / 欠拟合 / 发散，并出诊断图。

用法:
    python diagnose_training.py --run runs/school_v3
"""
import argparse
import csv as csvmod
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load(path):
    """直接读 csv，避免依赖 polars。"""
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csvmod.DictReader(f))
    data = {}
    for k in rows[0].keys():
        try:
            data[k.strip()] = [float(r[k]) for r in rows]
        except (ValueError, TypeError):
            data[k.strip()] = [0.0 for _ in rows]
    return data


def col(df, name):
    return df.get(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    csv = os.path.join(args.run, "results.csv")
    df = load(csv)
    ep = col(df, "epoch")
    tr_box, val_box = col(df, "train/box_loss"), col(df, "val/box_loss")
    tr_cls, val_cls = col(df, "train/cls_loss"), col(df, "val/cls_loss")
    m50, m5095 = col(df, "metrics/mAP50(B)"), col(df, "metrics/mAP50-95(B)")

    n = len(ep)
    best_i = max(range(len(m50)), key=lambda i: m50[i])
    best_m = float(m50[best_i])
    last_m = float(m50[-1])

    def mean(a, lo, hi):
        seg = [v for v in a[lo:hi] if v == v]  # 去掉 nan
        return float(sum(seg) / len(seg)) if seg else float("nan")

    gap_early = mean(val_box, 0, min(10, n)) - mean(tr_box, 0, min(10, n))
    gap_late = mean(val_box, max(0, n - 10), n) - mean(tr_box, max(0, n - 10), n)
    val_rise = mean(val_box, max(0, best_i), n) - mean(val_box, 0, max(1, best_i))
    train_dir = mean(tr_box, max(0, n - 5), n) - mean(tr_box, 0, 5)

    print("=" * 62)
    print("训练诊断:", args.run)
    print("=" * 62)
    print("总轮数        : %d" % n)
    print("最佳轮次      : epoch %d (mAP50 = %.3f)" % (int(ep[best_i]) + 1, best_m))
    print("最后一轮 mAP50: %.3f   (比最佳低 %.3f)" % (last_m, best_m - last_m))
    print("train/box_loss: 早期 %.2f -> 末期 %.2f (%+.2f)" %
          (mean(tr_box, 0, 5), mean(tr_box, max(0, n - 5), n), train_dir))
    print("val/box_loss  : 早期 %.2f -> 末期 %.2f (%+.2f)" %
          (mean(val_box, 0, 5), mean(val_box, max(0, n - 5), n),
           mean(val_box, max(0, n - 5), n) - mean(val_box, 0, 5)))
    print("验证-训练 差  : 早期 %+.2f -> 末期 %+.2f" % (gap_early, gap_late))
    print()

    verdict, reasons = [], []
    # 1) 发散：最佳轮次很早，之后 mAP 归零或暴跌
    if int(ep[best_i]) <= n * 0.3 and last_m < best_m * 0.5:
        verdict.append("训练发散")
        reasons.append("最佳结果出现在第 %d 轮，之后 mAP 从 %.2f 掉到 %.2f" %
                       (int(ep[best_i]) + 1, best_m, last_m))
    # 2) 过拟合：验证损失在最佳之后持续上升
    if n > 15 and val_rise > 0.15 * max(1e-6, mean(val_box, 0, max(1, best_i))):
        verdict.append("过拟合")
        reasons.append("最佳轮次之后 val/box_loss 上升 %.2f" % val_rise)
    # 3) 欠拟合：训练损失几乎没降，或末段仍在明显下降
    if abs(train_dir) < 0.15 * max(1e-6, mean(tr_box, 0, 5)):
        verdict.append("欠拟合")
        reasons.append("train/box_loss 几乎没下降（%+.2f）" % train_dir)
    if abs(gap_late) < 0.05 and best_m < 0.5:
        verdict.append("欠拟合")
        reasons.append("验证与训练损失几乎相等（差 %+.2f）且 mAP 偏低" % gap_late)
    # 4) 早停：最佳轮次远早于结束
    if n - int(ep[best_i]) > 20 and best_m > 0:
        reasons.append("第 %d 轮后连续 %d 轮无提升（早停触发）" %
                       (int(ep[best_i]) + 1, n - int(ep[best_i]) - 1))

    if not verdict:
        verdict = ["正常"]
        reasons.append("训练/验证损失同步下降，mAP 单调上升")

    print("判定:", " + ".join(dict.fromkeys(verdict)))
    for r in reasons:
        print("  -", r)
    print()
    print("参考阈值：")
    print("  过拟合  验证损失持续上升，训练损失还在降（gap 越拉越大）")
    print("  欠拟合  训练损失降不动，验证与训练几乎相等，mAP 又低")
    print("  发散    损失突然跳升、mAP 归零（多半是学习率太大）")

    # 画图
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0][0].plot(ep, tr_box, label="train")
    ax[0][0].plot(ep, val_box, label="val")
    ax[0][0].set_title("box_loss"); ax[0][0].legend(); ax[0][0].grid(alpha=.3)

    ax[0][1].plot(ep, tr_cls, label="train")
    ax[0][1].plot(ep, val_cls, label="val")
    ax[0][1].set_title("cls_loss"); ax[0][1].legend(); ax[0][1].grid(alpha=.3)

    ax[1][0].plot(ep, m50, label="mAP50")
    ax[1][0].plot(ep, m5095, label="mAP50-95")
    ax[1][0].axvline(ep[best_i], color="r", ls="--", lw=1, label="best")
    ax[1][0].set_title("mAP"); ax[1][0].legend(); ax[1][0].grid(alpha=.3)

    if gap_early is not None:
        gap = [(v - t) if (v == v and t == t) else float("nan")
               for v, t in zip(val_box, tr_box)]
        ax[1][1].plot(ep, gap, color="purple")
        ax[1][1].axhline(0, color="gray", lw=.8)
        ax[1][1].set_title("val - train box_loss（看是否越拉越大）")
        ax[1][1].grid(alpha=.3)

    fig.suptitle("训练诊断: %s" % os.path.basename(args.run))
    fig.tight_layout()
    out = args.out or os.path.join(args.run, "diagnose.png")
    fig.savefig(out, dpi=110)
    print("\n诊断图:", out)


if __name__ == "__main__":
    main()
