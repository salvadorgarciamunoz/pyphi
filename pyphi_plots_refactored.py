#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots for pyPhi

@author: Sal Garcia <sgarciam@ic.ac.uk> <salvadorgarciamunoz@gmail.com>

Refactored:
  - Extracted shared helpers: _get_lv_labels, _get_xvar_labels, _get_yvar_labels,
    _new_output_file, _make_bokeh_palette, _resolve_lpls_space, _mask_by_class
  - Fixed loop-variable collision in predvsobs (i → j)
  - Fixed output_file inside loop in mb_r2pb
  - Fixed 2-D y_ array in score_line (use .flatten())
  - Added unbound-variable guard in contributions_plot
  - Replaced False sentinels with None throughout
  - Replaced list(np.arange(...)+1) with range(1, n+1)
  - Replaced string concatenation with f-strings
  - Replaced .tolist() on ColumnDataSource inputs (Bokeh accepts ndarrays)
  - Replaced O(n*k) classification loops with pandas masking
  - Standardised colormap keyword args (alpha=1, bytes=True)
  - Replaced math.pi with np.pi
"""

from __future__ import annotations

import math  # kept only for backward-compat; internally we use np.pi
from datetime import datetime
from typing import Optional, Union

import matplotlib
import numpy as np
import pandas as pd
import pyphi as phi
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, LabelSet, Legend, Span
from bokeh.plotting import figure

__all__ = [
    "r2pv", "loadings", "loadings_map", "weighted_loadings", "vip",
    "score_scatter", "score_line", "diagnostics", "predvsobs",
    "contributions_plot", "mb_weights", "mb_r2pb", "mb_vip",
    "barplot", "lineplot", "plot_spectra", "scatter_with_labels",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _timestr() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def _new_output_file(prefix: str, title: str) -> None:
    output_file(f"{prefix}_{_timestr()}.html", title=title, mode="inline")


def _get_lv_labels(mvmobj: dict) -> list[str]:
    """Return list of LV / PC label strings."""
    A = mvmobj["T"].shape[1]
    prefix = "LV #" if "Q" in mvmobj else "PC #"
    return [f"{prefix}{a}" for a in range(1, A + 1)]


def _get_xvar_labels(mvmobj: dict) -> list[str]:
    if "varidX" in mvmobj:
        return mvmobj["varidX"]
    n = mvmobj["P"].shape[0]
    return [f"XVar #{i}" for i in range(1, n + 1)]


def _get_yvar_labels(mvmobj: dict) -> list[str]:
    if "varidY" in mvmobj:
        return mvmobj["varidY"]
    n = mvmobj["Q"].shape[0]
    return [f"YVar #{i}" for i in range(1, n + 1)]


def _make_bokeh_palette(n: int, cmap_name: str = "rainbow") -> list[str]:
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(np.linspace(0, 1, n), alpha=1, bytes=True)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgba[:, :3]]


def _resolve_lpls_space(mvmobj: dict, material, zspace: bool) -> dict:
    """
    Mutate a copy of mvmobj so that Ws / varidX / r2xpv point at the
    right sub-space for lpls / jrpls / tpls models.  Returns the copy.
    """
    obj = mvmobj.copy()
    t = obj["type"]
    if t == "lpls":
        obj["Ws"] = obj["Ss"]
    if (t in ("jrpls", "tpls")) and material is not None:
        idx = obj["materials"].index(material)
        obj["Ws"] = obj["Ssi"][idx]
        obj["varidX"] = obj["varidXi"][idx]
        if "r2xpvi" in obj:
            obj["r2xpv"] = obj["r2xpvi"][idx]
    elif t == "tpls" and zspace:
        obj["varidX"] = obj["varidZ"]
        if "r2zpv" in obj:
            obj["r2xpv"] = obj["r2zpv"]
    elif t in ("jrpls", "tpls") and material is None:
        obj["Ws"] = obj["Ss"]
    return obj


def _mask_by_class(
    classid_list: list,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    obs_ids: list[str],
    obs_nums: list[str],
    class_val,
) -> dict:
    """Return a ColumnDataSource-ready dict filtered to one class."""
    mask = np.array([c == class_val for c in classid_list])
    return dict(
        x=x_arr[mask].tolist(),
        y=y_arr[mask].tolist(),
        ObsID=[obs_ids[i] for i in np.where(mask)[0]],
        ObsNum=[obs_nums[i] for i in np.where(mask)[0]],
        Class=[class_val] * int(mask.sum()),
    )


def _add_origin_lines(p) -> None:
    p.renderers.extend([
        Span(location=0, dimension="height", line_color="black", line_width=2),
        Span(location=0, dimension="width",  line_color="black", line_width=2),
    ])


def _add_hline(p) -> None:
    p.renderers.append(
        Span(location=0, dimension="width", line_color="black", line_width=2)
    )


def _add_ci_ellipse(p, T_matrix: np.ndarray, mvmobj: dict, xd: int, yd: int) -> None:
    T1 = T_matrix[:, [xd - 1]]
    T2 = T_matrix[:, [yd - 1]]
    T_aux = np.hstack((T1, T2))
    st = (T_aux.T @ T_aux) / T_aux.shape[0]
    xd95, xd99, yd95p, yd95n, yd99p, yd99n = phi.scores_conf_int_calc(
        st, mvmobj["T"].shape[0]
    )
    p.line(xd95, yd95p, line_color="gold", line_dash="dashed")
    p.line(xd95, yd95n, line_color="gold", line_dash="dashed")
    p.line(xd99, yd99p, line_color="red",  line_dash="dashed")
    p.line(xd99, yd99n, line_color="red",  line_dash="dashed")


def _obs_ids_from_model(mvmobj: dict) -> list[str]:
    if "obsidX" in mvmobj:
        return mvmobj["obsidX"]
    return [f"Obs #{n}" for n in range(1, mvmobj["T"].shape[0] + 1)]


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def r2pv(
    mvm_obj: dict,
    *,
    plotwidth: int = 600,
    plotheight: int = 400,
    addtitle: str = "",
    material=None,
    zspace: bool = False,
) -> None:
    """R² per variable per component plots."""
    mvmobj = _resolve_lpls_space(mvm_obj, material, zspace)
    A = mvmobj["T"].shape[1]
    is_pls = "Q" in mvmobj
    yaxlbl = "Z" if (mvmobj["type"] == "tpls" and zspace) else "X"
    lv_labels = _get_lv_labels(mvmobj)
    XVar = _get_xvar_labels(mvmobj)

    r2pvX_dict: dict = {"XVar": XVar}
    for i, lbl in enumerate(lv_labels):
        r2pvX_dict[lbl] = mvmobj["r2xpv"][:, i]

    palette = _make_bokeh_palette(A)

    if is_pls:
        _new_output_file("r2xypv", f"R2{yaxlbl}YPV")
        YVar = _get_yvar_labels(mvmobj)
        r2pvY_dict: dict = {"YVar": YVar}
        for i, lbl in enumerate(lv_labels):
            r2pvY_dict[lbl] = mvmobj["r2ypv"][:, i]

        def _bar(x_range, title, source_dict, key, ylabel):
            p = figure(
                x_range=x_range, title=title,
                tools="save,box_zoom,xpan,hover,reset",
                tooltips=f"$name @{key}: @$name",
                width=plotwidth, height=plotheight,
            )
            v = p.vbar_stack(lv_labels, x=key, width=0.9, color=palette, source=source_dict)
            p.y_range.range_padding = 0.1
            p.ygrid.grid_line_color = None
            p.xgrid.grid_line_color = None
            p.axis.minor_tick_line_color = None
            p.outline_line_color = None
            p.yaxis.axis_label = ylabel
            p.xaxis.major_label_orientation = np.pi / 2
            p.add_layout(Legend(items=[(lbl, [v[i]]) for i, lbl in enumerate(lv_labels)]), "right")
            return p

        px = _bar(XVar, f"R2{yaxlbl} Per Variable {addtitle}", r2pvX_dict, "XVar", f"R2{yaxlbl}")
        py = _bar(YVar, f"R2Y Per Variable {addtitle}", r2pvY_dict, "YVar", "R2Y")
        show(column(px, py))
    else:
        _new_output_file("r2xpv", "R2XPV")
        p = figure(
            x_range=XVar, title=f"R2X Per Variable {addtitle}",
            tools="save,box_zoom,xpan,hover,reset",
            tooltips="$name @XVar: @$name",
            width=plotwidth, height=plotheight,
        )
        v = p.vbar_stack(lv_labels, x="XVar", width=0.9, color=palette, source=r2pvX_dict)
        p.y_range.range_padding = 0.1
        p.ygrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        p.yaxis.axis_label = "R2X"
        p.xaxis.major_label_orientation = np.pi / 2
        p.add_layout(Legend(items=[(lbl, [v[i]]) for i, lbl in enumerate(lv_labels)]), "right")
        show(p)


def loadings(
    mvm_obj: dict,
    *,
    plotwidth: int = 600,
    xgrid: bool = False,
    addtitle: str = "",
    material=None,
    zspace: bool = False,
) -> None:
    """Column plots of loadings.

    PLS models: one HTML, one bar plot per LV.  Each plot shows X loadings
    (blue) and Y loadings (red) on a shared categorical axis, separated by a
    vertical divider.  PCA models: one HTML, one bar plot per LV (X only).
    """
    mvmobj = _resolve_lpls_space(mvm_obj, material, zspace)
    t = mvmobj["type"]
    space_lbl = "Z" if (t == "tpls" and zspace) else "X"
    is_pls = "Q" in mvmobj
    lv_labels = _get_lv_labels(mvmobj)
    XVar = _get_xvar_labels(mvmobj)

    if t in ("lpls", "jrpls", "tpls"):
        loading_lbl = "Wz*" if (t == "tpls" and zspace) else "S*"
        X_coeff = mvmobj["Ws"]
    else:
        loading_lbl = "W*" if is_pls else "P"
        X_coeff = mvmobj["Ws"] if is_pls else mvmobj["P"]

    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"

    if not is_pls:
        # PCA: X-only, one figure per LV
        _new_output_file(f"Loadings_{space_lbl}_Space", f"{space_lbl} Loadings PCA")
        p_list = []
        for i, lbl in enumerate(lv_labels):
            src = ColumnDataSource(dict(x_=XVar, y_=X_coeff[:, i], names=XVar))
            p = figure(x_range=XVar,
                       title=f"{space_lbl} Space Loadings {lbl}{addtitle}",
                       tools=TOOLS, tooltips=[("Variable:", "@names")],
                       width=plotwidth)
            p.vbar(x="x_", top="y_", source=src, width=0.5, color="steelblue",
                   legend_label=loading_lbl)
            p.ygrid.grid_line_color = None
            p.xgrid.grid_line_color = "lightgray" if xgrid else None
            p.yaxis.axis_label = f"{loading_lbl} [{i+1}]"
            p.legend.location = "top_right"
            _add_hline(p)
            p.xaxis.major_label_orientation = np.pi / 2
            p_list.append(p)
        show(column(p_list))

    else:
        # PLS: combined X + Y on one axis per LV, single HTML
        YVar = _get_yvar_labels(mvmobj)
        combined_vars = XVar + YVar          # shared categorical axis
        n_x = len(XVar)
        divider_pos = XVar[-1]               # Span sits after the last X variable

        _new_output_file("Loadings_XY", f"{space_lbl}/Y Loadings PLS")
        p_list = []
        for i, lbl in enumerate(lv_labels):
            x_vals = list(X_coeff[:, i])
            y_vals = list(mvmobj["Q"][:, i])
            pad_x  = [np.nan] * len(YVar)   # no bar for Y positions in X series
            pad_y  = [np.nan] * len(XVar)   # no bar for X positions in Y series

            src = ColumnDataSource(dict(
                var      = combined_vars,
                x_load   = x_vals + pad_x,
                y_load   = pad_y  + y_vals,
                names    = combined_vars,
            ))
            TOOLTIPS_combined = [("Variable:", "@names")]
            p = figure(x_range=combined_vars,
                       title=f"{space_lbl}/Y Loadings {lbl}{addtitle}",
                       tools=TOOLS, tooltips=TOOLTIPS_combined,
                       width=plotwidth)

            bx = p.vbar(x="var", top="x_load", source=src, width=0.5,
                        color="steelblue", legend_label=f"{loading_lbl} (X)",
                        nan_policy="omit")
            by = p.vbar(x="var", top="y_load", source=src, width=0.5,
                        color="tomato", legend_label="Q (Y)",
                        nan_policy="omit")

            # Vertical divider between X and Y sections
            p.add_layout(Span(location=n_x - 0.5, dimension="height",
                              line_color="black", line_width=1,
                              line_dash="dashed"))
            p.ygrid.grid_line_color = None
            p.xgrid.grid_line_color = "lightgray" if xgrid else None
            p.yaxis.axis_label = f"{loading_lbl} / Q [{i+1}]"
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"
            _add_hline(p)
            p.xaxis.major_label_orientation = np.pi / 2
            p_list.append(p)
        show(column(p_list))


def loadings_map(
    mvm_obj: dict,
    dims: list[int],
    *,
    plotwidth: int = 600,
    addtitle: str = "",
    material=None,
    zspace: bool = False,
    textalpha: float = 0.75,
) -> None:
    """Scatter plot overlaying X and Y loadings."""
    mvmobj = _resolve_lpls_space(mvm_obj, material, zspace)
    is_pls = "Q" in mvmobj
    lv_labels = _get_lv_labels(mvmobj)
    XVar = _get_xvar_labels(mvmobj)

    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [("index", "$index"), ("(x,y)", "($x, $y)"), ("Variable:", "@names")]
    _new_output_file("Loadings_Map", "Loadings Map")

    def _norm(v):
        return v / np.max(np.abs(v))

    d0, d1 = dims[0] - 1, dims[1] - 1

    if is_pls:
        YVar = _get_yvar_labels(mvmobj)
        src_x = ColumnDataSource(dict(x=_norm(mvmobj["Ws"][:, d0]),
                                       y=_norm(mvmobj["Ws"][:, d1]), names=XVar))
        src_y = ColumnDataSource(dict(x=_norm(mvmobj["Q"][:, d0]),
                                       y=_norm(mvmobj["Q"][:, d1]), names=YVar))
        p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth,
                   title=f"Loadings Map LV[{dims[0]}] - LV[{dims[1]}] {addtitle}",
                   x_range=(-1.5, 1.5), y_range=(-1.5, 1.5))
        p.scatter("x", "y", source=src_x, size=10, color="darkblue")
        p.scatter("x", "y", source=src_y, size=10, color="red")
        for src, col in ((src_x, "darkgray"), (src_y, "darkgray")):
            p.add_layout(LabelSet(x="x", y="y", text="names", level="glyph",
                                   x_offset=5, y_offset=5, source=src,
                                   text_color=col, text_alpha=textalpha))
    else:
        src_x = ColumnDataSource(dict(x=mvmobj["P"][:, d0],
                                       y=mvmobj["P"][:, d1], names=XVar))
        p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth,
                   title=f"Loadings Map PC[{dims[0]}] - PC[{dims[1]}] {addtitle}",
                   x_range=(-1.5, 1.5), y_range=(-1.5, 1.5))
        p.scatter("x", "y", source=src_x, size=10, color="darkblue")
        p.add_layout(LabelSet(x="x", y="y", text="names", level="glyph",
                               x_offset=5, y_offset=5, source=src_x,
                               text_color="darkgray", text_alpha=textalpha))

    p.xaxis.axis_label = lv_labels[d0]
    p.yaxis.axis_label = lv_labels[d1]
    _add_origin_lines(p)
    show(p)


def weighted_loadings(
    mvm_obj: dict,
    *,
    plotwidth: int = 600,
    xgrid: bool = False,
    addtitle: str = "",
    material=None,
    zspace: bool = False,
) -> None:
    """Column plots of loadings weighted by r2x / r2y.

    PLS models: one HTML, one bar plot per LV.  Each plot shows
    W*×R²X (blue) and Q×R²Y (red) on a shared categorical axis,
    separated by a vertical divider.  PCA models: one HTML, X only.
    """
    mvmobj = _resolve_lpls_space(mvm_obj, material, zspace)
    t = mvmobj["type"]
    is_pls = "Q" in mvmobj
    space_lbl = "Z" if (t == "tpls" and zspace) else "X"
    loading_lbl = "Wz*" if (t == "tpls" and zspace) else (
        "S*" if t in ("lpls", "jrpls", "tpls") else "W*"
    )
    lv_labels = _get_lv_labels(mvmobj)
    XVar = _get_xvar_labels(mvmobj)
    X_coeff = mvmobj["Ws"] if is_pls else mvmobj["P"]

    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"

    if not is_pls:
        # PCA: X-only
        _new_output_file(f"WeightedLoadings_{space_lbl}_Space",
                         f"{space_lbl} Weighted Loadings PCA")
        p_list = []
        for i, lbl in enumerate(lv_labels):
            vals = mvmobj["r2xpv"][:, i] * X_coeff[:, i]
            src = ColumnDataSource(dict(x_=XVar, y_=vals, names=XVar))
            p = figure(x_range=XVar,
                       title=f"{space_lbl} Space Weighted Loadings {lbl}{addtitle}",
                       tools=TOOLS, tooltips=[("Variable:", "@names")],
                       width=plotwidth)
            p.vbar(x="x_", top="y_", source=src, width=0.5, color="steelblue",
                   legend_label=f"{loading_lbl}×R²{space_lbl}")
            p.ygrid.grid_line_color = None
            p.xgrid.grid_line_color = "lightgray" if xgrid else None
            p.yaxis.axis_label = f"{loading_lbl} × R²{space_lbl} [{i+1}]"
            p.legend.location = "top_right"
            _add_hline(p)
            p.xaxis.major_label_orientation = np.pi / 2
            p_list.append(p)
        show(column(p_list))

    else:
        # PLS: combined X + Y per LV, single HTML
        YVar = _get_yvar_labels(mvmobj)
        combined_vars = XVar + YVar
        n_x = len(XVar)

        _new_output_file("WeightedLoadings_XY",
                         f"{space_lbl}/Y Weighted Loadings PLS")
        p_list = []
        for i, lbl in enumerate(lv_labels):
            wx = list(mvmobj["r2xpv"][:, i] * X_coeff[:, i])
            wy = list(mvmobj["r2ypv"][:, i] * mvmobj["Q"][:, i])
            pad_x = [np.nan] * len(YVar)
            pad_y = [np.nan] * len(XVar)

            src = ColumnDataSource(dict(
                var    = combined_vars,
                x_wl   = wx + pad_x,
                y_wl   = pad_y + wy,
                names  = combined_vars,
            ))
            p = figure(x_range=combined_vars,
                       title=f"{space_lbl}/Y Weighted Loadings {lbl}{addtitle}",
                       tools=TOOLS, tooltips=[("Variable:", "@names")],
                       width=plotwidth)

            p.vbar(x="var", top="x_wl", source=src, width=0.5,
                   color="steelblue",
                   legend_label=f"{loading_lbl}×R²{space_lbl}",
                   nan_policy="omit")
            p.vbar(x="var", top="y_wl", source=src, width=0.5,
                   color="tomato",
                   legend_label="Q×R²Y",
                   nan_policy="omit")

            p.add_layout(Span(location=n_x - 0.5, dimension="height",
                              line_color="black", line_width=1,
                              line_dash="dashed"))
            p.ygrid.grid_line_color = None
            p.xgrid.grid_line_color = "lightgray" if xgrid else None
            p.yaxis.axis_label = f"{loading_lbl}×R²{space_lbl} / Q×R²Y [{i+1}]"
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"
            _add_hline(p)
            p.xaxis.major_label_orientation = np.pi / 2
            p_list.append(p)
        show(column(p_list))


def vip(
    mvm_obj: dict,
    *,
    plotwidth: int = 600,
    material=None,
    zspace: bool = False,
    addtitle: str = "",
) -> None:
    """Variable Importance in Projection plot."""
    mvmobj = _resolve_lpls_space(mvm_obj, material, zspace)
    if "Q" not in mvmobj:
        return

    XVar = _get_xvar_labels(mvmobj)
    vip_vals = np.sum(
        np.abs(mvmobj["Ws"] * np.tile(mvmobj["r2y"], (mvmobj["Ws"].shape[0], 1))),
        axis=1,
    )
    sort_idx = np.argsort(-vip_vals)
    sorted_vars = [XVar[i] for i in sort_idx]
    sorted_vip  = vip_vals[sort_idx]

    _new_output_file("VIP", "VIP Coefficient")
    src = ColumnDataSource(dict(x_=sorted_vars, y_=sorted_vip, names=sorted_vars))
    p = figure(x_range=sorted_vars, title=f"VIP {addtitle}",
               tools="save,box_zoom,pan,reset",
               tooltips=[("Variable", "@names")], width=plotwidth)
    p.vbar(x="x_", top="y_", source=src, width=0.5)
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = "Very Important to the Projection"
    p.xaxis.major_label_orientation = np.pi / 2
    show(p)


def _create_classid_(df: pd.DataFrame, column: str, *, nbins: int = 5) -> pd.DataFrame:
    """Internal: create a CLASSID dataframe from numeric values binned into nbins groups."""
    vals = df[column].values
    nan_mask = np.isnan(vals)
    valid = vals[~nan_mask]
    _, bin_edges = np.histogram(valid, bins=nbins)
    range_list = [
        f"{np.round(bin_edges[i], 3)} to {np.round(bin_edges[i+1], 3)}"
        for i in range(len(bin_edges) - 1)
    ]
    edges_ = bin_edges.copy()
    edges_[-1] += 0.1
    membership_valid = np.digitize(valid, edges_) - 1

    membership = []
    valid_counter = 0
    for is_nan in nan_mask:
        if is_nan:
            membership.append("Missing Value")
        else:
            membership.append(range_list[membership_valid[valid_counter]])
            valid_counter += 1

    out = df[[df.columns[0]]].copy()
    out[column] = membership
    return out


def score_scatter(
    mvm_obj: dict,
    xydim: list[int],
    *,
    CLASSID: Optional[pd.DataFrame] = None,
    colorby: Optional[str] = None,
    Xnew=None,
    add_ci: bool = False,
    add_labels: bool = False,
    add_legend: bool = True,
    legend_cols: int = 1,
    addtitle: str = "",
    plotwidth: int = 600,
    plotheight: int = 600,
    rscores: bool = False,
    material=None,
    marker_size: int = 7,
    nbins=None,
    include_model: bool = False,
) -> None:
    """Score scatter plot."""
    if nbins is not None and CLASSID is not None and colorby in CLASSID.columns.tolist():
        CLASSID = _create_classid_(CLASSID, colorby, nbins=nbins)

    mvmobj = mvm_obj.copy()
    if mvmobj["type"] in ("lpls", "jrpls", "tpls") and Xnew is not None:
        Xnew = None
        print("score_scatter does not take Xnew for jrpls or lpls for now")

    if Xnew is None:
        ObsID_ = _obs_ids_from_model(mvmobj)
        T_matrix = mvmobj["T"]

        if not rscores:
            if mvmobj["type"] == "lpls":
                ObsID_ = mvmobj["obsidR"]
            elif mvmobj["type"] in ("jrpls", "tpls"):
                ObsID_ = mvmobj["obsidRi"][0]
        else:
            if mvmobj["type"] == "lpls":
                ObsID_ = mvmobj["obsidX"]
                T_matrix = mvmobj["Rscores"]
            elif mvmobj["type"] in ("jrpls", "tpls"):
                if material is None:
                    ObsID_ = [o for sub in mvmobj["obsidXi"] for o in sub]
                    T_matrix = np.vstack(mvmobj["Rscores"])
                    classes = [
                        m for i, m in enumerate(mvmobj["materials"])
                        for _ in mvmobj["obsidXi"][i]
                    ]
                    CLASSID = pd.DataFrame({"obs": ObsID_, "material": classes})
                    colorby = "material"
                else:
                    idx = mvmobj["materials"].index(material)
                    ObsID_ = mvmobj["obsidXi"][idx]
                    T_matrix = mvmobj["Rscores"][idx]
    else:
        if isinstance(Xnew, np.ndarray):
            X_ = Xnew.copy()
            ObsID_ = [f"Obs #{n}" for n in range(1, Xnew.shape[0] + 1)]
        elif isinstance(Xnew, pd.DataFrame):
            X_ = Xnew.values[:, 1:].astype(float)
            ObsID_ = Xnew.values[:, 0].astype(str).tolist()
        pred = phi.pls_pred(X_, mvmobj) if "Q" in mvmobj else phi.pca_pred(X_, mvmobj)
        T_matrix = pred["Tnew"]

    if include_model:
        ObsID_model = _obs_ids_from_model(mvmobj)
        T_model = mvmobj["T"].copy()
        if CLASSID is None:
            source_col = ["Model"] * T_model.shape[0] + ["New"] * T_matrix.shape[0]
            ObsID_ = ObsID_model + ObsID_
            CLASSID = pd.DataFrame({"ObsID": ObsID_, "_Source_": source_col})
            colorby = "_Source_"
        else:
            src_model = pd.DataFrame({CLASSID.columns[0]: ObsID_model, colorby: ["Model"] * T_model.shape[0]})
            CLASSID = pd.concat([src_model, CLASSID], ignore_index=True)
            ObsID_ = ObsID_model + ObsID_
        T_matrix = np.vstack((T_model, T_matrix))

    ObsNum_ = [str(n) for n in range(1, len(ObsID_) + 1)]
    x_ = T_matrix[:, xydim[0] - 1]
    y_ = T_matrix[:, xydim[1] - 1]

    _new_output_file("Score_Scatter", f"Score Scatter t[{xydim[0]}] - t[{xydim[1]}]")
    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [("Obs #", "@ObsNum"), ("(x,y)", "($x, $y)"), ("Obs: ", "@ObsID")]
    ax_lbl = "r" if rscores else "t"

    p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth, height=plotheight,
               title=f"Score Scatter {ax_lbl}[{xydim[0]}] - {ax_lbl}[{xydim[1]}] {addtitle}",
               toolbar_location="above" if CLASSID is not None else "right")

    if CLASSID is None:
        src = ColumnDataSource(dict(x=x_, y=y_, ObsID=ObsID_, ObsNum=ObsNum_))
        p.scatter("x", "y", source=src, size=marker_size)
        if add_labels:
            p.add_layout(LabelSet(x="x", y="y", text="ObsID", level="glyph",
                                   x_offset=5, y_offset=5, source=src))
    else:
        Classes_ = phi.unique(CLASSID, colorby)
        if nbins is not None:
            # Sort numeric bin labels
            non_mv = [c for c in Classes_ if c != "Missing Value"]
            order = np.argsort([float(c.split()[0]) for c in non_mv])
            Classes_ = (["Missing Value"] if "Missing Value" in Classes_ else []) + \
                        [non_mv[i] for i in order]

        cmap_name = "viridis" if nbins is not None else "rainbow"
        if Classes_ and Classes_[0] in ("Model", "Missing Value"):
            # Regenerate n-1 colours so non-special classes keep full spread,
            # then prepend gray — matches original behaviour for both sentinels.
            inner = _make_bokeh_palette(len(Classes_) - 1, cmap_name)
            palette = ["#e1e1e1"] + inner
        else:
            palette = _make_bokeh_palette(len(Classes_), cmap_name)

        classid_arr = np.array(CLASSID[colorby].tolist())
        legend_it = []
        TOOLTIPS_cls = TOOLTIPS + [("Class:", "@Class")]
        p.tooltips = TOOLTIPS_cls

        for cls_val, color_ in zip(Classes_, palette):
            data = _mask_by_class(classid_arr, x_, y_, ObsID_, ObsNum_, cls_val)
            src = ColumnDataSource(data)
            c = p.scatter("x", "y", source=src, color=color_, size=marker_size)
            if add_labels:
                p.add_layout(LabelSet(x="x", y="y", text="ObsID", level="glyph",
                                       x_offset=5, y_offset=5, source=src))
            if add_legend:
                legend_it.append((str(cls_val), [c]))

        if add_legend:
            ipc = [int(np.round(len(legend_it) / legend_cols))] * legend_cols
            ipc[-1] = len(legend_it) - sum(ipc[:-1])
            offset = 0
            for chunk in ipc:
                leg = Legend(items=legend_it[offset:offset + chunk])
                p.add_layout(leg, "right")
                leg.click_policy = "hide"
                offset += chunk

    if add_ci:
        _add_ci_ellipse(p, T_matrix, mvmobj, xydim[0], xydim[1])

    p.xaxis.axis_label = f"{ax_lbl} [{xydim[0]}]"
    p.yaxis.axis_label = f"{ax_lbl} [{xydim[1]}]"
    _add_origin_lines(p)
    show(p)


def score_line(
    mvmobj: dict,
    dim,
    *,
    CLASSID: Optional[pd.DataFrame] = None,
    colorby: Optional[str] = None,
    Xnew=None,
    add_ci: bool = False,
    add_labels: bool = False,
    add_legend: bool = True,
    plotline: bool = True,
    plotwidth: int = 600,
    plotheight: int = 600,
) -> None:
    """Score line plot."""
    if not isinstance(dim, list):
        dim = [dim] if isinstance(dim, int) else list(dim)

    if Xnew is None:
        ObsID_ = _obs_ids_from_model(mvmobj)
        T_matrix = mvmobj["T"]
    else:
        if isinstance(Xnew, np.ndarray):
            X_ = Xnew.copy()
            ObsID_ = [f"Obs #{n}" for n in range(1, Xnew.shape[0] + 1)]
        elif isinstance(Xnew, pd.DataFrame):
            X_ = Xnew.values[:, 1:].astype(float)
            ObsID_ = Xnew.values[:, 0].astype(str).tolist()
        pred = phi.pls_pred(X_, mvmobj) if "Q" in mvmobj else phi.pca_pred(X_, mvmobj)
        T_matrix = pred["Tnew"]

    ObsNum_ = [f"Obs #{n}" for n in range(1, len(ObsID_) + 1)]
    y_ = T_matrix[:, dim[0] - 1].flatten()  # Fix: was 2-D (n,1)
    x_ = list(range(1, len(y_) + 1))

    _new_output_file("Score_Line", f"Score Line t[{dim[0]}]")
    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [("Obs#", "@ObsNum"), ("(x,y)", "($x, $y)"), ("Obs: ", "@ObsID")]

    def _add_ci_lines(p):
        lim95, lim99 = phi.single_score_conf_int(mvmobj["T"][:, [dim[0] - 1]])
        for lim, col in ((lim95, "gold"), (lim99, "red")):
            p.line(x_,  lim, line_color=col, line_dash="dashed")
            p.line(x_, -lim, line_color=col, line_dash="dashed")

    if CLASSID is None:
        src = ColumnDataSource(dict(x=x_, y=y_, ObsID=ObsID_, ObsNum=ObsNum_))
        p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth, height=plotheight,
                   title=f"Score Line t[{dim[0]}]")
        p.scatter("x", "y", source=src, size=10)
        if plotline:
            p.line("x", "y", source=src)
        if add_ci:
            _add_ci_lines(p)
        if add_labels:
            p.add_layout(LabelSet(x="x", y="y", text="ObsID", level="glyph",
                                   x_offset=5, y_offset=5, source=src))
    else:
        Classes_ = phi.unique(CLASSID, colorby)
        palette = _make_bokeh_palette(len(Classes_))
        classid_arr = np.array(CLASSID[colorby].tolist())
        TOOLTIPS_cls = TOOLTIPS + [("Class:", "@Class")]

        p = figure(tools=TOOLS, tooltips=TOOLTIPS_cls, toolbar_location="above",
                   width=plotwidth, height=plotheight, title=f"Score Line t[{dim[0]}]")
        legend_it = []
        x_arr = np.array(x_)

        for cls_val, color_ in zip(Classes_, palette):
            data = _mask_by_class(classid_arr, x_arr, y_, ObsID_, ObsNum_, cls_val)
            src = ColumnDataSource(data)
            c = p.scatter("x", "y", source=src, color=color_, size=10)
            glyphs = [c]
            if plotline:
                c1 = p.line("x", "y", source=src, color=color_)
                glyphs.append(c1)
            if add_labels:
                p.add_layout(LabelSet(x="x", y="y", text="ObsID", level="glyph",
                                       x_offset=5, y_offset=5, source=src))
            if add_legend:
                legend_it.append((str(cls_val), glyphs))

        if add_ci:
            _add_ci_lines(p)
        if add_legend:
            leg = Legend(items=legend_it, location="top_right")
            p.add_layout(leg, "right")
            leg.click_policy = "hide"

    p.xaxis.axis_label = "Observation"
    p.yaxis.axis_label = f"t [{dim[0]}]"
    show(p)


def diagnostics(
    mvmobj: dict,
    *,
    Xnew=None,
    Ynew=None,
    score_plot_xydim=None,
    plotwidth: int = 600,
    ht2_logscale: bool = False,
    spe_logscale: bool = False,
) -> None:
    """Hotelling's T² and SPE diagnostic plots."""
    add_score_plot = score_plot_xydim is not None

    if Xnew is None:
        ObsID_ = _obs_ids_from_model(mvmobj)
        Obs_num = np.arange(mvmobj["T"].shape[0]) + 1
        t2_    = mvmobj["T2"].copy()
        spex_  = mvmobj["speX"].copy()
        spey_  = mvmobj.get("speY") if "Q" in mvmobj else None

        if add_score_plot:
            t_x = mvmobj["T"][:, [score_plot_xydim[0] - 1]]
            t_y = mvmobj["T"][:, [score_plot_xydim[1] - 1]]
        else:
            t_x = t_y = None
    else:
        if isinstance(Xnew, np.ndarray):
            X_ = Xnew
            ObsID_ = [f"Obs #{n}" for n in range(1, Xnew.shape[0] + 1)]
        elif isinstance(Xnew, pd.DataFrame):
            X_ = Xnew.values[:, 1:].astype(float)
            ObsID_ = Xnew.values[:, 0].astype(str).tolist()

        t2_ = phi.hott2(mvmobj, Xnew=Xnew)
        Obs_num = np.arange(t2_.shape[0]) + 1

        if "Q" in mvmobj and Ynew is not None:
            spex_, spey_ = phi.spe(mvmobj, Xnew, Ynew=Ynew)
        else:
            spex_ = phi.spe(mvmobj, Xnew)
            spey_ = None

        if add_score_plot:
            pred = phi.pls_pred(X_, mvmobj) if "Q" in mvmobj else phi.pca_pred(X_, mvmobj)
            T_matrix = pred["Tnew"]
            t_x = T_matrix[:, [score_plot_xydim[0] - 1]]
            t_y = T_matrix[:, [score_plot_xydim[1] - 1]]
        else:
            t_x = t_y = None

    if ht2_logscale:
        t2_ = np.log10(t2_)
    if spe_logscale:
        spex_ = np.log10(spex_)

    ObsNum_ = [f"Obs #{n}" for n in range(1, len(ObsID_) + 1)]
    src_dict = dict(x=Obs_num, ObsID=ObsID_, ObsNum=ObsNum_, t2=t2_, spex=spex_)
    if spey_ is not None:
        src_dict["spey"] = spey_
    if add_score_plot:
        src_dict["tx"] = t_x
        src_dict["ty"] = t_y
    source = ColumnDataSource(src_dict)

    TOOLS = "save,wheel_zoom,box_zoom,reset,lasso_select"
    TOOLTIPS = [("Obs #", "@x"), ("(x,y)", "($x, $y)"), ("Obs: ", "@ObsID")]
    _new_output_file("Diagnostics", "Diagnostics")

    def _lim(val, log):
        return np.log10(val) if log else val

    p_list = []

    # T2
    p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth, title="Hotelling's T2")
    p.scatter("x", "t2", source=source)
    p.line([0, Obs_num[-1]], [_lim(mvmobj["T2_lim95"], ht2_logscale)] * 2, line_color="gold")
    p.line([0, Obs_num[-1]], [_lim(mvmobj["T2_lim99"], ht2_logscale)] * 2, line_color="red")
    p.xaxis.axis_label = "Observation sequence"
    p.yaxis.axis_label = "HT2"
    p_list.append(p)

    # SPE X
    p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth, title="SPE X")
    p.scatter("x", "spex", source=source)
    p.line([0, Obs_num[-1]], [_lim(mvmobj["speX_lim95"], spe_logscale)] * 2, line_color="gold")
    p.line([0, Obs_num[-1]], [_lim(mvmobj["speX_lim99"], spe_logscale)] * 2, line_color="red")
    p.xaxis.axis_label = "Observation sequence"
    p.yaxis.axis_label = "SPE X-Space"
    p_list.append(p)

    # Outlier map
    p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth, title="Outlier Map")
    p.scatter("t2", "spex", source=source)
    p.renderers.extend([
        Span(location=_lim(mvmobj["T2_lim99"],  ht2_logscale), dimension="height", line_color="red", line_width=1),
        Span(location=_lim(mvmobj["speX_lim99"], spe_logscale), dimension="width",  line_color="red", line_width=1),
    ])
    p.xaxis.axis_label = "Hotelling's T2"
    p.yaxis.axis_label = "SPE X-Space"
    p_list.append(p)

    # SPE Y
    if "Q" in mvmobj and spey_ is not None:
        p = figure(tools=TOOLS, tooltips=TOOLTIPS, height=400, title="SPE Y")
        p.scatter("x", "spey", source=source, size=10)
        p.line([0, Obs_num[-1]], [mvmobj["speY_lim95"]] * 2, line_color="gold")
        p.line([0, Obs_num[-1]], [mvmobj["speY_lim99"]] * 2, line_color="red")
        p.xaxis.axis_label = "Observation sequence"
        p.yaxis.axis_label = "SPE Y-Space"
        p_list.append(p)

    # Score scatter
    if add_score_plot:
        p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth, title="Score Scatter")
        p.scatter("tx", "ty", source=source, size=10)
        _add_ci_ellipse(p, mvmobj["T"], mvmobj, score_plot_xydim[0], score_plot_xydim[1])
        p.xaxis.axis_label = f"t [{score_plot_xydim[0]}]"
        p.yaxis.axis_label = f"t [{score_plot_xydim[1]}]"
        _add_origin_lines(p)
        p_list.append(p)

    show(column(p_list))


def predvsobs(
    mvmobj: dict,
    X,
    Y,
    *,
    CLASSID: Optional[pd.DataFrame] = None,
    colorby: Optional[str] = None,
    x_space: bool = False,
) -> None:
    """Observed vs predicted values."""
    if isinstance(X, np.ndarray):
        X_ = X.copy()
        ObsID_ = [f"Obs #{n}" for n in range(1, X.shape[0] + 1)]
    elif isinstance(X, pd.DataFrame):
        X_ = X.values[:, 1:].astype(float)
        ObsID_ = X.values[:, 0].astype(str).tolist()
    elif isinstance(X, dict):
        parts = list(X.values())
        ObsID_ = parts[0].values[:, 0].astype(str).tolist()
        X_ = pd.concat([parts[0]] + [p.iloc[:, 1:] for p in parts[1:]], axis=1)

    XVar = _get_xvar_labels(mvmobj)

    if isinstance(Y, np.ndarray):
        Y_ = Y.copy()
    elif isinstance(Y, pd.DataFrame):
        Y_ = Y.values[:, 1:].astype(float)

    if "Q" in mvmobj:
        pred = phi.pls_pred(X_, mvmobj)
        yhat = pred["Yhat"]
        xhat = pred["Xhat"] if x_space else None
        YVar = _get_yvar_labels(mvmobj)
    else:
        x_space = True
        pred = phi.pca_pred(X_, mvmobj)
        xhat = pred["Xhat"]
        yhat = None

    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [("index", "$index"), ("(x,y)", "($x, $y)"), ("Obs: ", "@ObsID")]
    _new_output_file("ObsvsPred", "ObsvsPred")

    def _scatter_panel(obs, pred_vals, var_name, plot_counter):
        mn = np.nanmin([np.nanmin(obs), np.nanmin(pred_vals)])
        mx = np.nanmax([np.nanmax(obs), np.nanmax(pred_vals)])
        p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=600, height=600,
                   title=var_name, x_range=(mn, mx), y_range=(mn, mx))
        p.line([mn, mx], [mn, mx], line_color="cyan", line_dash="dashed")
        p.xaxis.axis_label = "Observed"
        p.yaxis.axis_label = "Predicted"
        return p

    p_list = []
    plot_counter = 0

    if CLASSID is None:
        all_panels = []
        if yhat is not None:
            for i in range(Y_.shape[1]):
                p = _scatter_panel(Y_[:, i], yhat[:, i], YVar[i], plot_counter)
                src = ColumnDataSource(dict(x=Y_[:, i], y=yhat[:, i], ObsID=ObsID_))
                p.scatter("x", "y", source=src, size=7, color="darkblue")
                all_panels.append(p)
        if x_space and xhat is not None:
            for i in range(X_.shape[1]):  # Fix: was using outer `i` shadowed by inner loop
                p = _scatter_panel(X_[:, i], xhat[:, i], XVar[i], plot_counter)
                src = ColumnDataSource(dict(x=X_[:, i], y=xhat[:, i], ObsID=ObsID_))
                p.scatter("x", "y", source=src, size=10, color="darkblue")
                all_panels.append(p)
        show(column(all_panels))
    else:
        Classes_ = phi.unique(CLASSID, colorby)
        palette = _make_bokeh_palette(len(Classes_))
        classid_ = list(CLASSID[colorby])

        def _add_classified(obs_col, pred_col, var_name):
            p = _scatter_panel(obs_col, pred_col, var_name, 0)
            for cls_val, color_ in zip(Classes_, palette):
                mask = np.array([c == cls_val for c in classid_])
                valid = mask & ~np.isnan(obs_col)
                src = ColumnDataSource(dict(
                    x=obs_col[valid], y=pred_col[valid],
                    ObsID=[ObsID_[j] for j in np.where(valid)[0]],
                    Class=[cls_val] * valid.sum(),
                ))
                p.scatter("x", "y", source=src, color=color_, legend_label=str(cls_val))
            p.legend.click_policy = "hide"
            p.legend.location = "top_left"
            return p

        all_panels = []
        if yhat is not None:
            for i in range(Y_.shape[1]):
                all_panels.append(_add_classified(Y_[:, i], yhat[:, i], YVar[i]))
        if x_space and xhat is not None:
            for i in range(X_.shape[1]):  # Fix: was `i` collision
                all_panels.append(_add_classified(X_[:, i], xhat[:, i], XVar[i]))
        show(column(all_panels))


def contributions_plot(
    mvmobj: dict,
    X,
    cont_type: str,
    *,
    Y=None,
    from_obs=None,
    to_obs=None,
    lv_space=None,
    plotwidth: int = 800,
    plotheight: int = 600,
    xgrid: bool = False,
) -> None:
    """Plot contributions to diagnostics."""
    # Flatten multi-block dicts to DataFrames
    def _dict_to_df(d):
        parts = list(d.values())
        return pd.concat([parts[0]] + [p.iloc[:, 1:] for p in parts[1:]], axis=1)

    if isinstance(X, dict):
        X = _dict_to_df(X)
    if isinstance(Y, dict):
        Y = _dict_to_df(Y)

    # Resolve obs indices
    def _resolve_obs(obs, obs_ids):
        if obs is None:
            return None
        if isinstance(obs, str):
            return obs_ids.index(obs)
        if isinstance(obs, int):
            return obs
        if isinstance(obs, list):
            if isinstance(obs[0], str):
                return [obs_ids.index(o) for o in obs]
            return obs.copy()
        return None

    if isinstance(X, pd.DataFrame):
        obs_ids = X.values[:, 0].tolist()
        to_obs_   = _resolve_obs(to_obs, obs_ids)
        from_obs_ = _resolve_obs(from_obs, obs_ids)
    else:
        to_obs_   = to_obs
        from_obs_ = from_obs

    if to_obs_ is None:
        print("contributions_plot: to_obs is required.")
        return

    if cont_type == "scores":
        Y = None

    if Y is None:
        Xconts = phi.contributions(mvmobj, X, cont_type, Y=None,
                                   from_obs=from_obs_, to_obs=to_obs_, lv_space=lv_space)
        Yconts = None
    elif "Q" in mvmobj and cont_type == "spe":
        Xconts, Yconts = phi.contributions(mvmobj, X, cont_type, Y=Y,
                                            from_obs=from_obs_, to_obs=to_obs_, lv_space=lv_space)
    else:
        print("contributions_plot: unsupported combination of Y and cont_type.")
        return

    XVar = _get_xvar_labels(mvmobj)

    def _obs_txt(obs, prefix):
        if obs is None:
            return ""
        if isinstance(obs, list):
            return f"{prefix}{', '.join(map(str, obs))}"
        return f"{prefix}{obs}"

    from_txt = _obs_txt(from_obs, " from obs: ")
    to_txt   = _obs_txt(to_obs,   ", to obs: ")

    _new_output_file("Contributions", "Contributions")
    TOOLTIPS = [("Variable", "@names")]

    p_list = []
    src = ColumnDataSource(dict(x_=XVar, y_=Xconts[0], names=XVar))
    p = figure(x_range=XVar, height=plotheight, width=plotwidth,
               title=f"Contributions Plot{from_txt}{to_txt}",
               tools="save,box_zoom,pan,reset", tooltips=TOOLTIPS)
    p.vbar(x="x_", top="y_", source=src, width=0.5)
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "lightgray" if xgrid else None
    p.yaxis.axis_label = f"Contributions to {cont_type}"
    _add_hline(p)
    p.xaxis.major_label_orientation = np.pi / 2
    p_list.append(p)

    if Yconts is not None:
        YVar = _get_yvar_labels(mvmobj)
        src_y = ColumnDataSource(dict(x_=YVar, y_=Yconts[0], names=YVar))
        p = figure(x_range=YVar, height=plotheight, width=plotwidth,
                   title="Contributions Plot", tools="save,box_zoom,pan,reset")
        p.vbar(x="x_", top="y_", source=src_y, width=0.5)
        p.ygrid.grid_line_color = None
        p.xgrid.grid_line_color = "lightgray" if xgrid else None
        p.yaxis.axis_label = f"Contributions to {cont_type}"
        _add_hline(p)
        p.xaxis.major_label_orientation = np.pi / 2
        p_list.append(p)

    show(column(p_list))


def mb_weights(mvmobj: dict, *, plotwidth: int = 600, plotheight: int = 400) -> None:
    """Super weights for Multi-block models."""
    lv_labels = _get_lv_labels(mvmobj)
    XVar = mvmobj["Xblocknames"]
    _new_output_file("blockweights", "Block Weights")
    p_list = []
    for i, lbl in enumerate(lv_labels):
        src = ColumnDataSource(dict(x_=XVar, y_=mvmobj["Wt"][:, i], names=XVar))
        p = figure(x_range=XVar, title=f"Block weights for MBPLS {lbl}",
                   tools="save,box_zoom,hover,reset",
                   tooltips=[("Var:", "@x_")], width=plotwidth, height=plotheight)
        p.vbar(x="x_", top="y_", source=src, width=0.5)
        p.y_range.range_padding = 0.1
        p.ygrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        p.yaxis.axis_label = f"Wt [{i+1}]"
        p.xaxis.major_label_orientation = np.pi / 2
        _add_hline(p)
        p_list.append(p)
    show(column(p_list))


def mb_r2pb(mvmobj: dict, *, plotwidth: int = 600, plotheight: int = 400) -> None:
    """R² per block for Multi-block models."""
    A = mvmobj["T"].shape[1]
    lv_labels = _get_lv_labels(mvmobj)
    XVar = mvmobj["Xblocknames"]
    palette = _make_bokeh_palette(A)
    r2_dict: dict = {"XVar": XVar}
    for i, lbl in enumerate(lv_labels):
        r2_dict[lbl] = mvmobj["r2pbX"][:, i]

    _new_output_file("r2perblock", "R2 per Block")  # Fix: was inside the loop
    p = figure(x_range=XVar, title="R2 per Block for MBPLS",
               tools="save,box_zoom,hover,reset",
               tooltips="$name @XVar: @$name",
               width=plotwidth, height=plotheight)
    p.vbar_stack(lv_labels, x="XVar", width=0.9, color=palette, source=r2_dict)
    p.y_range.range_padding = 0.1
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.yaxis.axis_label = "R2 per Block per LV"
    p.xaxis.major_label_orientation = np.pi / 2
    show(p)


def mb_vip(mvmobj: dict, *, plotwidth: int = 600, plotheight: int = 400) -> None:
    """VIP per block for Multi-block models."""
    A = mvmobj["T"].shape[1]
    XVar = mvmobj["Xblocknames"]
    Wt = mvmobj["Wt"]
    r2y = mvmobj["r2y"]

    vip_vals = np.sum(Wt * r2y if A == 1 else
                      np.column_stack([Wt[:, a] * r2y[a] for a in range(A)]),
                      axis=1)
    order = np.argsort(-vip_vals)
    XVar_sorted = [XVar[i] for i in order]
    vip_sorted  = vip_vals[order]

    _new_output_file("blockvip", "Block VIP")
    src = ColumnDataSource(dict(x_=XVar_sorted, y_=vip_sorted, names=XVar_sorted))
    p = figure(x_range=XVar_sorted, title="Block VIP for MBPLS",
               tools="save,box_zoom,hover,reset",
               tooltips=[("Block:", "@x_")], width=plotwidth, height=plotheight)
    p.vbar(x="x_", top="y_", source=src, width=0.5)
    p.y_range.range_padding = 0.1
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.yaxis.axis_label = "Block VIP"
    p.xaxis.major_label_orientation = np.pi / 2
    _add_hline(p)
    show(p)


def barplot(
    yheights,
    xtick_labels: list[str],
    *,
    plotwidth: int = 600,
    plotheight: int = 600,
    addtitle: str = "",
    xlabel: str = "",
    ylabel: str = "",
    tabtitle: str = "Bar Plot",
) -> None:
    """Generic bar plot."""
    _new_output_file("BarPlot", tabtitle)
    src = ColumnDataSource(dict(x_=xtick_labels, y_=yheights, names=xtick_labels))
    p = figure(x_range=xtick_labels, title=addtitle,
               tools="save,box_zoom,pan,reset",
               tooltips=[("Variable", "@names")], width=plotwidth)
    p.vbar(x="x_", top="y_", source=src, width=0.5)
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = ylabel
    p.xaxis.axis_label = xlabel
    p.xaxis.major_label_orientation = np.pi / 2
    show(p)


def lineplot(
    X,
    *,
    ids_2_include=None,
    x_axis=None,
    plot_title: str = "Main Title",
    tab_title: str = "Tab Title",
    xaxis_label: str = "X-axis",
    yaxis_label: str = "",
    plotheight: int = 400,
    plotwidth: int = 600,
    legend_cols: int = 1,
    linecolor: str = "blue",
    linewidth: int = 2,
    add_marker: bool = False,
    individual_plots: bool = False,
    add_legend: bool = True,
    markercolor: str = "darkblue",
    markersize: int = 10,
    fill_alpha: float = 0.2,
    line_alpha: float = 0.4,
    ncx_x_col=None,
    ncx_y_col=None,
    ncx_id_col=None,
    CLASSID: Optional[pd.DataFrame] = None,
    colorby: Optional[str] = None,
    yaxis_log: bool = False,
) -> None:
    """Plot columns of a DataFrame or a list of DataFrames with Bokeh."""
    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [("Obs #", "@ObsNum"), ("(x,y)", "($x, $y)"), ("ID: ", "@ColID")]
    _new_output_file("LinePlot", tab_title)

    def _make_figure(title_):
        kw = dict(y_axis_type="log") if yaxis_log else {}
        return figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth,
                      height=plotheight, title=title_, **kw)

    def _get_x(n):
        if x_axis is None:
            return list(range(1, n + 1))
        return list(x_axis) if isinstance(x_axis, np.ndarray) else x_axis

    def _add_legend_layout(p, legend_it):
        ipc = [int(np.round(len(legend_it) / legend_cols))] * legend_cols
        ipc[-1] = len(legend_it) - sum(ipc[:-1])
        offset = 0
        for chunk in ipc:
            leg = Legend(items=legend_it[offset:offset + chunk])
            p.add_layout(leg, "right")
            leg.click_policy = "hide"
            offset += chunk

    # --- Common x-axis (X is a DataFrame) ---
    if isinstance(X, pd.DataFrame):
        if ids_2_include is None:
            ids_2_include = X.columns.tolist()
        elif isinstance(ids_2_include, str):
            ids_2_include = [ids_2_include]

        if individual_plots:
            p_list = []
            for k, col_name in enumerate(ids_2_include):
                y_ = X[col_name].values
                x_ = _get_x(X.shape[0])
                src = ColumnDataSource(dict(x=x_, y=y_,
                                            ColID=[col_name] * X.shape[0], ObsNum=x_))
                p = _make_figure(plot_title if k == 0 else "")
                line_kw = dict(line_color=linecolor, color=markercolor,
                               line_width=linewidth, line_alpha=line_alpha)
                if add_legend:
                    p.line("x", "y", source=src, legend_label=col_name, **line_kw)
                else:
                    p.line("x", "y", source=src, **line_kw)
                if add_marker:
                    p.scatter("x", "y", source=src, color=markercolor,
                              size=markersize, fill_alpha=fill_alpha)
                _add_hline(p)
                p.xaxis.axis_label = xaxis_label
                p.yaxis.axis_label = yaxis_label
                p_list.append(p)
            show(column(p_list))

        elif CLASSID is None:
            palette = _make_bokeh_palette(len(ids_2_include))
            p = _make_figure(plot_title)
            legend_it = []
            for col_name, color_ in zip(ids_2_include, palette):
                y_ = X[col_name].values
                x_ = _get_x(X.shape[0])
                src = ColumnDataSource(dict(x=x_, y=y_ if y_.ndim == 1 else y_,
                                            ColID=[col_name] * len(x_), ObsNum=x_))
                if y_.ndim == 1:
                    g = p.line("x", "y", source=src, line_color=color_,
                               line_width=linewidth, line_alpha=line_alpha, color=color_)
                    glyphs = [g]
                    if add_marker:
                        gm = p.scatter("x", "y", source=src, color=color_,
                                       size=markersize, fill_alpha=fill_alpha)
                        glyphs.append(gm)
                    legend_it.append((col_name, glyphs))
                else:
                    for ci in range(y_.shape[1]):
                        src_ = ColumnDataSource(dict(x=x_, y=y_[:, ci],
                                                     ColID=[col_name] * len(x_), ObsNum=x_))
                        g = p.line("x", "y", source=src_, line_color=color_,
                                   line_width=linewidth, line_alpha=line_alpha)
                        glyphs = [g]
                        if add_marker:
                            gm = p.scatter("x", "y", source=src_, color=color_,
                                           size=markersize, fill_alpha=fill_alpha)
                            glyphs.append(gm)
                        legend_it.append((col_name, glyphs))
                _add_hline(p)
            p.xaxis.axis_label = xaxis_label
            p.yaxis.axis_label = yaxis_label
            if add_legend:
                _add_legend_layout(p, legend_it)
            show(p)

        else:  # CLASSID provided
            classes = phi.unique(CLASSID, colorby)
            palette = _make_bokeh_palette(len(classes))
            p = _make_figure(plot_title)
            legend_it = []
            for cls_val, color_ in zip(classes, palette):
                cols = CLASSID[CLASSID.columns[0]][CLASSID[colorby] == cls_val].values.tolist()
                leg_glyphs = []
                for col_name in cols:
                    y_ = X[col_name].values
                    x_ = _get_x(X.shape[0])
                    if y_.ndim == 1:
                        src = ColumnDataSource(dict(x=x_, y=y_,
                                                    ColID=[col_name] * len(x_), ObsNum=x_))
                        g = p.line("x", "y", source=src, line_color=color_,
                                   line_width=linewidth, line_alpha=line_alpha, color=color_)
                        leg_glyphs.append(g)
                        if add_marker:
                            gm = p.scatter("x", "y", source=src, color=color_,
                                           size=markersize, fill_alpha=fill_alpha)
                            leg_glyphs.append(gm)
                    else:
                        for ci in range(y_.shape[1]):
                            src_ = ColumnDataSource(dict(x=x_, y=y_[:, ci],
                                                         ColID=[col_name] * len(x_), ObsNum=x_))
                            g = p.line("x", "y", source=src_, line_color=color_,
                                       line_width=linewidth, line_alpha=line_alpha)
                            leg_glyphs.append(g)
                            if add_marker:
                                gm = p.scatter("x", "y", source=src_, color=color_,
                                               size=markersize, fill_alpha=fill_alpha)
                                leg_glyphs.append(gm)
                legend_it.append((str(cls_val), leg_glyphs))
            _add_hline(p)
            p.xaxis.axis_label = xaxis_label
            p.yaxis.axis_label = yaxis_label
            if add_legend:
                _add_legend_layout(p, legend_it)
            show(p)

    # --- Non-common x-axis (X is a list of DataFrames) ---
    elif (isinstance(X, list) and ncx_y_col is not None
          and ncx_x_col is not None and ncx_id_col is not None):

        if ids_2_include is None:
            ids_2_include = [df[ncx_id_col].values[0] for df in X]
        elif isinstance(ids_2_include, str):
            ids_2_include = [ids_2_include]

        id_to_df = {df[ncx_id_col].values[0]: df for df in X}

        if individual_plots:
            p_list = []
            for k, col_name in enumerate(ids_2_include):
                df = id_to_df[col_name]
                x_ = df[ncx_x_col].values
                y_ = df[ncx_y_col].values
                src = ColumnDataSource(dict(x=x_, y=y_,
                                            ColID=[col_name] * len(x_), ObsNum=x_))
                p = _make_figure(plot_title if k == 0 else "")
                lkw = dict(line_color=linecolor, color=markercolor,
                           line_width=linewidth, line_alpha=line_alpha)
                if add_legend:
                    p.line("x", "y", source=src, legend_label=col_name, **lkw)
                else:
                    p.line("x", "y", source=src, **lkw)
                if add_marker:
                    p.scatter("x", "y", source=src, color=markercolor,
                              size=markersize, fill_alpha=fill_alpha)
                _add_hline(p)
                p.xaxis.axis_label = ncx_x_col
                p.yaxis.axis_label = ncx_y_col
                p_list.append(p)
            show(column(p_list))
        else:
            palette = _make_bokeh_palette(len(ids_2_include))
            p = _make_figure(plot_title)
            legend_it = []
            for col_name, color_ in zip(ids_2_include, palette):
                df = id_to_df[col_name]
                x_ = df[ncx_x_col].values
                y_ = df[ncx_y_col].values
                src = ColumnDataSource(dict(x=x_, y=y_,
                                            ColID=[col_name] * len(x_), ObsNum=x_))
                g = p.line("x", "y", source=src, line_color=color_,
                           line_width=linewidth, line_alpha=line_alpha, color=color_)
                glyphs = [g]
                if add_marker:
                    gm = p.scatter("x", "y", source=src, color=color_,
                                   size=markersize, fill_alpha=fill_alpha)
                    glyphs.append(gm)
                legend_it.append((col_name, glyphs))
                _add_hline(p)
            p.xaxis.axis_label = xaxis_label
            p.yaxis.axis_label = yaxis_label
            if add_legend:
                _add_legend_layout(p, legend_it)
            show(p)


def plot_spectra(
    X,
    *,
    xaxis=None,
    plot_title: str = "Main Title",
    tab_title: str = "Tab Title",
    xaxis_label: str = "X-axis",
    yaxis_label: str = "Y-axis",
    linecolor: str = "blue",
    linewidth: int = 2,
) -> None:
    """Plot spectra (one line per row)."""
    if isinstance(X, pd.DataFrame):
        x = np.array(X.columns[1:].tolist()).reshape(1, -1)
        y = X.values[:, 1:].astype(float)
    elif isinstance(X, np.ndarray):
        y = X.copy()
        if xaxis is None:
            x = np.arange(X.shape[1]).reshape(1, -1)
        else:
            x = np.array(xaxis).reshape(1, -1)

    _new_output_file("Spectra", tab_title)
    p = figure(title=plot_title)
    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = yaxis_label
    p.multi_line(x.tolist() * y.shape[0], y.tolist(),
                 line_color=linecolor, line_width=linewidth)
    show(p)


def scatter_with_labels(
    x,
    y,
    *,
    xlabel: str = "X var",
    ylabel: str = "Y var",
    labels=None,
    tabtitle: str = "Scatter Plot",
    plottitle: str = "Scatter",
    legend_cols: int = 1,
    CLASSID: Optional[pd.DataFrame] = None,
    colorby: Optional[str] = None,
    plotwidth: int = 600,
    plotheight: int = 600,
    markercolor: str = "darkblue",
    markersize: int = 10,
    fill_alpha: float = 0.2,
    line_alpha: float = 0.4,
) -> None:
    """Generic scatter plot with optional labels and class coloring."""
    if labels is None:
        labels = [f"Obs {i}" for i in range(len(x))]

    _new_output_file("Scatter_Plot", tabtitle)
    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [("index", "$index"), ("(x,y)", "($x, $y)"), ("Property", "@names")]

    p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth,
               height=plotheight, title=plottitle)

    if CLASSID is None or colorby is None:
        src = ColumnDataSource(dict(x=x, y=y, names=labels))
        p.scatter("x", "y", source=src, size=markersize, color=markercolor,
                  fill_alpha=fill_alpha, line_alpha=line_alpha)
    else:
        Classes_ = phi.unique(CLASSID, colorby)
        palette = _make_bokeh_palette(len(Classes_))
        x_arr = np.array(x)
        y_arr = np.array(y)
        labels_arr = np.array(labels)
        class_arr = CLASSID[colorby].values

        legend_it = []
        ipc = [int(np.round(len(Classes_) / legend_cols))] * legend_cols
        ipc[-1] = len(Classes_) - sum(ipc[:-1])

        for cls_val, color_ in zip(Classes_, palette):
            mask = class_arr == cls_val
            src = ColumnDataSource(dict(x=x_arr[mask], y=y_arr[mask], names=labels_arr[mask]))
            g = p.scatter("x", "y", source=src, color=color_, size=markersize,
                          fill_alpha=fill_alpha, line_alpha=line_alpha)
            legend_it.append((str(cls_val), [g]))

        offset = 0
        for chunk in ipc:
            leg = Legend(items=legend_it[offset:offset + chunk])
            p.add_layout(leg, "right")
            leg.click_policy = "hide"
            offset += chunk

    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    _add_origin_lines(p)
    show(p)
