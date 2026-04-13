import numpy as np
import matplotlib.pyplot as plt


def plot_ssc_q_diagnostic(
    time,
    Q,
    SSC,
    Q_flag,
    SSC_flag,
    ssc_q_bounds,
    station_id,
    station_name,
    out_png,
):
    """
    Create and save a station-level SSC-Q diagnostic plot.
    """
    if ssc_q_bounds is None:
        return

    time = np.asarray(time)
    Q = np.asarray(Q, dtype=float)
    SSC = np.asarray(SSC, dtype=float)
    Q_flag = np.asarray(Q_flag)
    SSC_flag = np.asarray(SSC_flag)

    valid = np.isfinite(Q) & np.isfinite(SSC) & (Q > 0) & (SSC > 0)
    if valid.sum() < 5:
        return

    logQ = np.log10(Q[valid])
    logSSC = np.log10(SSC[valid])

    coef = ssc_q_bounds["coef"]
    logSSC_pred = coef[0] * logQ + coef[1]
    resid = logSSC - logSSC_pred

    good = (Q_flag[valid] == 0) & (SSC_flag[valid] == 0)
    suspect = (SSC_flag[valid] == 2)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )

    ax = axes[0]
    ax.scatter(logQ[good], logSSC[good], s=20, c="tab:blue", alpha=0.7, label="Good")
    ax.scatter(logQ[suspect], logSSC[suspect], s=20, c="tab:red", alpha=0.7, label="Suspect")

    x_line = np.linspace(logQ.min(), logQ.max(), 200)
    y_mid = coef[0] * x_line + coef[1]
    y_low = y_mid + ssc_q_bounds["lower"]
    y_up = y_mid + ssc_q_bounds["upper"]

    ax.plot(x_line, y_mid, "k-", lw=2, label="Median trend")
    ax.plot(x_line, y_low, "k--", lw=1)
    ax.plot(x_line, y_up, "k--", lw=1)

    ax.set_xlabel("log10(Q) [m3/s]")
    ax.set_ylabel("log10(SSC) [mg/L]")
    ax.legend(frameon=False)
    ax.set_title("SSC-Q diagnostic: {0} ({1})".format(station_name, station_id))

    ax2 = axes[1]
    ax2.scatter(time[valid][good], resid[good], s=15, c="tab:blue", alpha=0.7)
    ax2.scatter(time[valid][suspect], resid[suspect], s=15, c="tab:red", alpha=0.7)

    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(ssc_q_bounds["lower"], color="k", ls="--")
    ax2.axhline(ssc_q_bounds["upper"], color="k", ls="--")

    ax2.set_ylabel("Residual (log SSC)")
    ax2.set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
