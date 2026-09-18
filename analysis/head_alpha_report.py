"""Conservative report rendering for the two registered inference scans."""
import json
import numpy as np
import pandas as pd

from analysis.head_alpha_preflight import HOSTS, write_json


def finish_report(output, rows, correlations, smoke=False):
    alpha_results={h:json.loads((output/h/'alpha/alpha_crossfit.json').read_text()) for h in HOSTS}
    alpha_gate=not smoke and all(r['delta']>0 and r['ci95'][0][2]>0 and r['stable_within_one_step'] for r in alpha_results.values())
    cross_host=[]
    for a,b in zip(rows[HOSTS[0]]['head'][1:],rows[HOSTS[1]]['head'][1:]):
        assert a['id']==b['id']
        cross_host.append(dict(id=a['id'],gwrp_delta_pp=a['delta_pp'],all_product_delta_pp=b['delta_pp'],
                               both_positive=a['delta_pp']>0 and b['delta_pp']>0,
                               both_lower_ci_positive=a['delta_ci95_lo_pp']>0 and b['delta_ci95_lo_pp']>0))
    pd.DataFrame(cross_host).to_csv(output/'head_cross_host.csv',index=False)
    head_gate=not smoke and any(r['both_lower_ci_positive'] for r in cross_host)
    topm=[]
    for a,b in zip(rows[HOSTS[0]]['topm'][1:],rows[HOSTS[1]]['topm'][1:]):
        assert a['id']==b['id']
        topm.append(dict(id=a['id'],gwrp_delta_pp=a['delta_pp'],all_product_delta_pp=b['delta_pp'],
                         both_positive=a['delta_pp']>0 and b['delta_pp']>0))
    pd.DataFrame(topm).to_csv(output/'topm_cross_host.csv',index=False)
    control=[]
    for host in HOSTS:
        for raw,norm in zip(rows[host]['control'][1::2],rows[host]['control'][2::2]):
            control.append(dict(host=host,layer=raw['layer'],head=raw['head'],
                                 rownorm_delta_pp=norm['fixed_mean_iou_percent']-raw['fixed_mean_iou_percent']))
    pd.DataFrame(control).to_csv(output/'rownorm_control.csv',index=False)
    decision=dict(smoke_only=smoke,alpha_gate=alpha_gate,head_gate_both_CI_positive=head_gate,
                  topm_both_hosts_improve=not smoke and any(r['both_positive'] for r in topm),
                  no_followup_authorized=True,
                  caution='Head maxima and best-m choices are in-sample sensitivity results; cross-host sign is not multiplicity correction. Alpha bootstrap repeats selection within fixed halves.')
    write_json(output/'decision.json',decision)
    alpha_lines=['# Alpha inference scan', '',
                 'All forwards/aggregation FP32; autocast OFF, TF32 OFF. Frozen epoch45 seed0 checkpoints; no training.',
                 'Native scales1,.75,1.25 + flip/minbound448, GT image-label gating, class min-max; fixed threshold.45.',
                 'Exact native reference is unclamped sqrt(ReLU(M)*a); alpha interior uses eps1e-8, endpoints exact.',
                 'The clamped alpha.5 point and exact native are reported separately. Best-threshold values are diagnostic.',
                 'Main crossfit statistic: size-weighted held-out-half dataset mIoU; alpha selection repeated in each5000 paired image bootstrap.', '',
                 '| Host | alpha_A / alpha_B | Native held-out mean | Crossfit | Delta pp [95% CI] | Stable |',
                 '|---|---|---:|---:|---|---|']
    for host in HOSTS:
        r=alpha_results[host]
        alpha_lines.append(f'| {host} | {r["selected_alphas"]} | {100*r["reference"]:.4f} | {100*r["point"]:.4f} | {100*r["delta"]:+.4f} [{100*r["ci95"][0][2]:+.4f}, {100*r["ci95"][1][2]:+.4f}] | {r["stable_within_one_step"]} |')
        central=[x['fixed_mean_iou_percent'] for x in rows[host]['alpha'][1:] if .4<=x['alpha']<=.6]
        best=max(rows[host]['alpha'][1:],key=lambda r:r['fixed_mean_iou_percent'])
        alpha_lines += ['',f'{host}: in-sample diagnostic best alpha={best["alpha"]}; central[.4,.6] range={np.ptp(central):.4f}pp.',
                        'Threshold cross-fitting is diagnostic only, in `'+host+'/alpha/threshold_crossfit_diagnostic.json`; it does not replace the prespecified.45 ranking.']
    alpha_lines += ['',f'Registered alpha gate: {alpha_gate}. No new queue authorized.',
                    'An endpoint optimum is evidence about these frozen CAM readouts, not proof that a training branch is unnecessary.']
    (output/'ALPHA_SWEEP_REPORT.md').write_text('\n'.join(alpha_lines)+'\n')
    head_lines=['# Head-axis inference scan','',
                'FP32, autocast OFF, TF32 OFF; only C2P head changes. P2P remains the native sum of all-layer head means.',
                '72 fixed cells plus exact native; row-normalized best/median/worst are controls, not selected methods.',
                'Gamma excludes single-label images. Unlabeled ranks use VOCval1449 image labels, not semantic masks, before head mIoU is measured.',
                'Small kappa=compact support; small gamma=more distinct positive-class maps. Both rankings and ties are prespecified.', '',
                '| Host | Native | Max cell | Max mIoU | Delta pp [95% CI] | Range pp |','|---|---:|---|---:|---|---:|']
    for host in HOSTS:
        head=rows[host]['head'][1:]
        best=max(head,key=lambda r:r['fixed_mean_iou_percent'])
        head_lines.append(f'| {host} | {rows[host]["head"][0]["fixed_mean_iou_percent"]:.4f} | {best["id"]} | {best["fixed_mean_iou_percent"]:.4f} | {best["delta_pp"]:+.4f} [{best["delta_ci95_lo_pp"]:+.4f}, {best["delta_ci95_hi_pp"]:+.4f}] | {np.ptp([r["fixed_mean_iou_percent"] for r in head]):.4f} |')
    head_lines += ['',pd.DataFrame(correlations).to_csv(index=False),
                   'Descriptive correlations across72 related cells are not72 independent observations. No causal interpretation or head-selection validity follows from rho alone.',
                   'The stricter reported positive gate requires the SAME cell to have lower CI>0 in BOTH hosts. Per-cell signs are retained in head_cross_host.csv.',
                   f'Strict replicated head gate: {head_gate}. Cross-host sign replication does not control72-comparison familywise error.',
                   'Top-m uses no mIoU ordering. Choosing best m after this sweep is still in-sample model selection; curves are not an independent test.',
                   'Row normalization may change view-relative weights before multi-scale summation and is not necessarily canceled by final min-max; epsilon also breaks exact scale invariance.',
                   'See rownorm_control.csv, topm_cross_host.csv, head_heatmap.svg and topm_curves.svg.']
    (output/'HEAD_AXIS_REPORT.md').write_text('\n'.join(head_lines)+'\n')
    (output/'HEAD_ALPHA_REPORT.md').write_text(
        '# Head / alpha inference-only experiment\n\nFP32 throughout; autocast and TF32 OFF. No model training or source checkpoint changes.\n\n'
        +('SMOKE ONLY; not full experiment results.\n\n' if smoke else 'Full VOC CAM train1464 and unlabeled val1449 completed for both seed0 hosts.\n\n')
        +'Read ALPHA_SWEEP_REPORT.md and HEAD_AXIS_REPORT.md. Exact configurations, folds, commands, source hashes and Git SHA are in adjacent JSON/CSV files.\n'
        +'Only compact confusion statistics and diagnostic summaries are retained; no per-image CAM or full attention dumps.\n'
        +'Native, alpha, head, rownorm and top-m comparisons are in each host/stage/comparison.csv.\n'
        +'The optional joint slice, trainable head weighting, extra seeds and all training are NOT part of this queue.\n')
    (output/'NEXT_EXPERIMENT_DECISION.md').write_text('# Registered gates — no automatic continuation\n\n'+json.dumps(decision,indent=2)+'\n')
