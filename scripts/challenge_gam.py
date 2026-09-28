"""Fixed descriptor GAM for the local DEC competition; predicts log_k directly."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import numpy as np
from pygam import LinearGAM, l, s
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


DESCRIPTORS = ['mw', 'tpsa', 'logp', 'rings']
CONTINUOUS = ['temperature_K', 'conc_native', *['mix_' + name for name in DESCRIPTORS]]
MODELS = ['constant', 'conditions_and_salt', 'descriptor_gam']
INPUT_FILES = ['train.csv', 'test.csv', 'sample_submission.csv', 'solvent_properties.csv', 'metaData.csv']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path, required):
    with path.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty input')
    return rows


@lru_cache(maxsize=None)
def canonical_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    return Chem.MolToSmiles(molecule)


@lru_cache(maxsize=None)
def molecule_descriptors(smiles):
    molecule = Chem.MolFromSmiles(canonical_smiles(smiles))
    return {'mw': float(Descriptors.MolWt(molecule)),
            'tpsa': float(rdMolDescriptors.CalcTPSA(molecule)),
            'logp': float(Crippen.MolLogP(molecule)),
            'rings': float(rdMolDescriptors.CalcNumRings(molecule))}


def mixture_descriptors(smiles, fractions):
    fractions = np.asarray(fractions, dtype=float)
    if (not smiles or len(smiles) != len(fractions) or not np.isfinite(fractions).all()
            or (fractions < 0).any() or not np.isclose(fractions.sum(), 1, rtol=0, atol=1e-8)):
        raise ValueError('Mole fractions must align with SMILES, be nonnegative and sum to one')
    components = [molecule_descriptors(item) for item in smiles]
    return {'mix_' + key: float(sum(x * d[key] for x, d in zip(fractions, components)))
            for key in DESCRIPTORS}


def prepare_rows(rows, properties, is_train):
    seen_ids = set()
    for row in rows:
        context = f"{'train' if is_train else 'test'}.csv id={row.get('id')}"
        try:
            if not row['id'] or row['id'] in seen_ids:
                raise ValueError('empty or duplicate ID')
            seen_ids.add(row['id'])
            names = row['solvents'].split(';')
            smiles = row['solvent_smiles'].split(';')
            fractions = list(map(float, row['solvent_fracs_mol'].split(';')))
            if len(names) != len(smiles) or len(set(names)) != len(names):
                raise ValueError('unaligned or duplicate solvent names')
            if ('DEC' in names) == is_train:
                raise ValueError('DEC must be absent in train and present in test')
            for name, structure in zip(names, smiles):
                if name not in properties or canonical_smiles(structure) != canonical_smiles(properties[name]):
                    raise ValueError(f'solvent structure disagrees with properties: {name}')
            row.update(mixture_descriptors(smiles, fractions))
            for feature in ['temperature_K', 'conc_native']:
                row[feature] = float(row[feature])
                if not np.isfinite(row[feature]):
                    raise ValueError(f'non-finite {feature}')
            if row['temperature_K'] <= 0 or row['conc_native'] < 0:
                raise ValueError('invalid temperature or concentration')
            if row['conc_unit'] not in ('mol/L', 'mol/kg') or not row['salt_name']:
                raise ValueError('invalid concentration unit or empty salt')
            if is_train:
                row['log_k'] = float(row['log_k'])
                if not np.isfinite(row['log_k']) or row['subset'] not in ('train', 'val') or not row['source_doi']:
                    raise ValueError('invalid target, subset or DOI')
            row['_formulation'] = (row['salt_name'], tuple(sorted(zip(names, fractions))),
                                   row['conc_native'], row['conc_unit'])
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'{context}: {error}') from error
    return rows


def encode_rows(rows, salts, with_descriptors=True):
    continuous = CONTINUOUS if with_descriptors else CONTINUOUS[:2]
    names = continuous + ['unit_molL'] + ['salt::' + salt for salt in salts]
    matrix = np.array([[*[float(row[name]) for name in continuous],
                        float(row['conc_unit'] == 'mol/L'),
                        *[float(row['salt_name'] == salt) for salt in salts]] for row in rows])
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite model input')
    return matrix, names


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Metric inputs must have matching shapes and finite values')
    errors = actual - predicted
    trimmed = actual >= -3
    return {'n': len(actual),
            'rmse': float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
            'mae': float(np.mean(np.abs(errors))) if len(errors) else None,
            'median_absolute_error': float(np.median(np.abs(errors))) if len(errors) else None,
            'p90_absolute_error': float(np.quantile(np.abs(errors), .9)) if len(errors) else None,
            'bias': float(errors.mean()) if len(errors) else None,
            'trimmed_rmse': float(np.sqrt(np.mean(errors[trimmed] ** 2))) if trimmed.any() else None,
            'trimmed_n': int(trimmed.sum()), 'excluded_from_trimmed': int((~trimmed).sum())}


def fit_predict(fitting, validation, model):
    target = np.array([row['log_k'] for row in fitting])
    if model == 'constant':
        value = float(target.mean())
        return np.full(len(validation), value), {'n_fit': len(fitting), 'constant_log_k': value}
    salts = sorted({row['salt_name'] for row in fitting})
    descriptors = model == 'descriptor_gam'
    x, names = encode_rows(fitting, salts, descriptors)
    xv, _ = encode_rows(validation, salts, descriptors)
    count = 6 if descriptors else 2
    means, scales = x[:, :count].mean(axis=0), x[:, :count].std(axis=0)
    scales[scales == 0] = 1
    x[:, :count] = (x[:, :count] - means) / scales
    xv[:, :count] = (xv[:, :count] - means) / scales
    terms = s(0, n_splines=5, spline_order=3, lam=.6) + s(1, n_splines=5, spline_order=3, lam=.6)
    for index in range(2, len(names)):
        terms += l(index, lam=1, penalties='l2')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gam = LinearGAM(terms).fit(x, target)
        prediction = gam.predict(xv)
    if not np.isfinite(prediction).all() or not np.isfinite(gam.coef_).all():
        raise RuntimeError('Non-finite GAM result')
    if gam.logs_.get('diffs', [float('inf')])[-1] >= gam.tol:
        raise RuntimeError('GAM did not converge')
    return prediction, {'n_fit': len(fitting), 'features': names, 'salts': salts,
                        'continuous_means': means.tolist(), 'continuous_scales': scales.tolist(),
                        'coefficient_count': len(gam.coef_), 'effective_degrees_of_freedom': float(gam.statistics_['edof']),
                        'coefficients': gam.coef_.tolist(), 'warnings': [str(w.message) for w in caught]}


def evaluate_check(fitting, validation, model, check):
    prediction, fit_info = fit_predict(fitting, validation, model)
    actual = np.array([r['log_k'] for r in validation])
    score = metrics(actual, prediction)
    salts = {r['salt_name'] for r in fitting}
    seen = np.array([r['salt_name'] in salts for r in validation])
    score['seen_salts'] = metrics(actual[seen], prediction[seen])
    score['unseen_salts'] = metrics(actual[~seen], prediction[~seen])
    score['unseen_salt_names'] = sorted({r['salt_name'] for r in validation} - salts)
    for field, key in [('salt_name', 'by_salt'), ('source_doi', 'by_doi')]:
        score[key] = {}
        for value in sorted({r[field] for r in validation}):
            mask = np.array([r[field] == value for r in validation])
            score[key][value] = metrics(actual[mask], prediction[mask])
    records = [{'check': check, 'model': model, 'id': row['id'], 'salt': row['salt_name'],
                'doi': row['source_doi'], 'log_k': row['log_k'], 'predicted_log_k': float(pred),
                'residual': float(row['log_k'] - pred), 'unseen_salt': row['salt_name'] not in salts}
               for row, pred in zip(validation, prediction)]
    return score, records, fit_info


def range_summary(fitting, validation):
    result = {}
    for feature in CONTINUOUS:
        fit = np.array([r[feature] for r in fitting])
        val = np.array([r[feature] for r in validation])
        result[feature] = {'fit_min': float(fit.min()), 'fit_max': float(fit.max()),
                           'validation_min': float(val.min()), 'validation_max': float(val.max()),
                           'outside_fit_range': int(((val < fit.min()) | (val > fit.max())).sum())}
    return result


def make_plots(output, predictions, rows_by_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for check in sorted({r['check'] for r in predictions}):
        rows = [r for r in predictions if r['check'] == check and r['model'] == 'descriptor_gam']
        y = np.array([r['log_k'] for r in rows])
        p = np.array([r['predicted_log_k'] for r in rows])
        residual = y - p
        salts = sorted({r['salt'] for r in rows})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
        for axis, mask, title in [(axes[0], np.ones(len(rows), dtype=bool), 'All validation rows'),
                                   (axes[1], y >= -3, 'View restricted to true log_k >= -3')]:
            for index, salt in enumerate(salts):
                selected = mask & np.array([r['salt'] == salt for r in rows])
                axis.scatter(y[selected], p[selected], s=10, alpha=.5, color=f'C{index % 10}', label=salt)
            lo, hi = min(y[mask].min(), p[mask].min()), max(y[mask].max(), p[mask].max())
            axis.plot([lo, hi], [lo, hi], '--', color='black', linewidth=1)
            axis.set(xlabel='Measured log10 conductivity', ylabel='Predicted log10 conductivity', title=title)
        axes[1].legend(fontsize=7, loc='best')
        fig.suptitle(f'{check}: descriptor GAM')
        fig.savefig(output / f'{check}_predictions.png', dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
        for axis, feature in zip(axes.flat, CONTINUOUS):
            values = np.array([rows_by_id[r['id']][feature] for r in rows])
            for index, salt in enumerate(salts):
                mask = np.array([r['salt'] == salt for r in rows])
                axis.scatter(values[mask], residual[mask], s=8, alpha=.4, color=f'C{index % 10}', label=salt)
            axis.axhline(0, linestyle='--', color='black', linewidth=1)
            axis.set(xlabel=feature, ylabel='Measured - predicted log_k')
        axes.flat[-1].legend(fontsize=7, loc='best')
        fig.suptitle(f'{check}: residuals (all validation rows)')
        fig.savefig(output / f'{check}_residuals.png', dpi=140)
        plt.close(fig)


def write_report(output, results, summary, submission, fit_info):
    lines = ['# First competition GAM', '',
             'Target and submission: log_k = log10(conductivity in mS/cm). No target clipping or inverse transform.', '',
             'The fixed GAM uses temperature and concentration smooths, a unit indicator, penalized salt offsets, '
             'and four mole-fraction-weighted RDKit solvent descriptors. DEC is represented by its structure, not a learned solvent-name coefficient.', '',
             '## Validation', '', '| Check | Model | RMSE | Trimmed RMSE | MAE | Median AE | P90 AE | Bias |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for check, models in results.items():
        for model, score in models.items():
            values = [score[k] for k in ['rmse', 'trimmed_rmse', 'mae', 'median_absolute_error', 'p90_absolute_error', 'bias']]
            lines.append(f'| {check} | {model} | ' + ' | '.join(f'{v:.4f}' if v is not None else 'n/a' for v in values) + ' |')
    lines += ['', 'Trimmed metrics include only true log_k >= -3; full metrics and training retain every row. '
              'Counts and per-salt/publication errors are in metrics.json.', '',
              '## Unseen-solvent checks', '', '| Model | Mean RMSE | SD across checks | Worst RMSE |',
              '| --- | ---: | ---: | ---: |']
    for model, value in summary.items():
        lines.append(f"| {model} | {value['mean_rmse']:.4f} | {value['std_rmse']:.4f} | {value['worst_rmse']:.4f} |")
    lines += ['', 'Two-check sample SD is descriptive spread, not a confidence interval. The supplied validation split is scored separately.', '',
              '## Descriptor contribution', '']
    for check, models in results.items():
        descriptor = models['descriptor_gam']
        for reference in ['constant', 'conditions_and_salt']:
            delta = descriptor['rmse'] - models[reference]['rmse']
            lines.append(f'- {check}: descriptor RMSE minus {reference} RMSE = {delta:+.4f} (negative is better).')
        lines.append(f"- {check}: unseen validation salts: {', '.join(descriptor['unseen_salt_names']) or 'none'}; "
                     f"{descriptor['unseen_salts']['n']} rows. Seen-salt-only RMSE: {descriptor['seen_salts']['rmse']:.4f}.")
    lines += ['', '## Scope and limitations', '',
              '- DMC and EMC checks exclude those solvents entirely from their fits, but their mixtures differ from the DEC test distribution.',
              '- Some salts also disappear from those fits. Their predictions use shared effects with zero salt-specific adjustment.',
              '- Mixture descriptor averages lose structural detail. Native concentration plus a unit offset does not perform a physical molarity/molality conversion.',
              '- The four descriptors have linear effects in this initial model; no interactions or model tuning were performed.',
              '- These are local validation results. Hidden DEC test labels were not available or recovered from the original dataset.', '',
              '## Final submission', '',
              f"The fixed descriptor GAM was refitted on all {fit_info['n_fit']:,} training rows. "
              f"It has {fit_info['coefficient_count']} coefficients and effective degrees of freedom {fit_info['effective_degrees_of_freedom']:.2f}.", '',
              f"submission.csv contains {len(submission):,} rows in sample-submission order, with columns id,log_k. "
              f"Prediction range: {min(r['log_k'] for r in submission):.4f} to {max(r['log_k'] for r in submission):.4f}. "
              'No observed test score is claimed. The model was not switched based on validation results.', '',
              '## Figures', '']
    for check in results:
        lines += [f'### {check}', '', f'![Predictions](diagnostics/{check}_predictions.png)', '',
                  f'![Residuals](diagnostics/{check}_residuals.png)', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(data_dir, output):
    required = ['id', 'salt_name', 'solvents', 'solvent_smiles', 'solvent_fracs_mol',
                'temperature_K', 'conc_native', 'conc_unit']
    solvent_rows = read_csv(data_dir / 'solvent_properties.csv', ['name', 'smiles'])
    properties = {r['name']: r['smiles'] for r in solvent_rows}
    if len(properties) != len(solvent_rows):
        raise ValueError('Duplicate solvent names in properties')
    train = prepare_rows(read_csv(data_dir / 'train.csv', required + ['log_k', 'subset', 'source_doi']), properties, True)
    test = prepare_rows(read_csv(data_dir / 'test.csv', required), properties, False)
    sample = read_csv(data_dir / 'sample_submission.csv', ['id', 'log_k'])
    if list(sample[0]) != ['id', 'log_k'] or len({r['id'] for r in sample}) != len(sample):
        raise ValueError('Invalid sample-submission schema or duplicate IDs')
    if {r['id'] for r in sample} != {r['id'] for r in test}:
        raise ValueError('Sample and test ID sets differ')
    if {r['id'] for r in train} & {r['id'] for r in test}:
        raise ValueError('Train and test IDs overlap')
    if {r['salt_name'] for r in test} - {r['salt_name'] for r in train}:
        raise ValueError('Final test contains unknown salts')
    development = [r for r in train if r['subset'] == 'train']
    validation = [r for r in train if r['subset'] == 'val']
    if not development or not validation:
        raise ValueError('Both supplied subsets must be nonempty')
    if {r['_formulation'] for r in development} & {r['_formulation'] for r in validation}:
        raise ValueError('Exact formulation crosses supplied split')
    checks = {'supplied': (development, validation)}
    for solvent in ['DMC', 'EMC']:
        fitting = [r for r in development if solvent not in r['solvents'].split(';')]
        held = [r for r in development if solvent in r['solvents'].split(';')]
        if not fitting or not held:
            raise ValueError(f'Empty {solvent} fit or validation split')
        checks[solvent] = (fitting, held)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = output / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    spec = {'target': 'log_k', 'target_units': 'log10(mS/cm)',
            'continuous_features': CONTINUOUS, 'unit_indicator': 'mol/L=1, mol/kg=0',
            'salt_encoding': 'one-hot from fitting rows; unknown salt all zero',
            'descriptor_functions': ['Descriptors.MolWt', 'rdMolDescriptors.CalcTPSA', 'Crippen.MolLogP', 'rdMolDescriptors.CalcNumRings'],
            'mixture_weighting': 'supplied mole fractions', 'terms': 's(T)+s(c)+l(unit)+salt offsets+four linear mixture descriptors',
            'n_splines': 5, 'spline_order': 3, 'smooth_lam': .6, 'linear_lam': 1.0, 'intercept': True,
            'continuous_scaling': 'fit-row mean and population SD; constant SD replaced with 1'}
    write_json(output / 'model_spec.json', spec)
    write_csv(output / 'solvent_descriptors.csv', [{'name': r['name'], 'smiles': canonical_smiles(r['smiles']),
               **molecule_descriptors(r['smiles'])} for r in solvent_rows])
    write_csv(output / 'splits.csv', [{'check': name, 'id': r['id'], 'partition': partition}
              for name, parts in checks.items() for partition, rows in zip(['fit', 'validation'], parts) for r in rows])
    provenance = {'input_sha256': {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in INPUT_FILES},
                  'versions': {name: version(name) for name in ['numpy', 'pygam', 'rdkit', 'matplotlib']},
                  'rows': {'train': len(train), 'test': len(test)}, 'target': spec['target'], 'target_units': spec['target_units'],
                  'split_rules': {'supplied': 'subset column', 'DMC': 'all DMC mixtures in subset=train withheld',
                                  'EMC': 'all EMC mixtures in subset=train withheld'},
                  'check_sizes': {name: {'fit': len(a), 'validation': len(b)} for name, (a, b) in checks.items()},
                  'chemical_validation_overlap': len({r['id'] for r in checks['DMC'][1]} & {r['id'] for r in checks['EMC'][1]}),
                  'fits': {}}
    all_predictions, results, ranges = [], {}, {}
    for check, (fitting, held) in checks.items():
        results[check] = {}
        ranges[check] = range_summary(fitting, held)
        for model in MODELS:
            score, predictions, fit_info = evaluate_check(fitting, held, model, check)
            results[check][model] = score
            provenance['fits'][check + '/' + model] = fit_info
            all_predictions.extend(predictions)
            print(f"{check:8} {model:20} RMSE={score['rmse']:.5f} MAE={score['mae']:.5f}", flush=True)
    summary = {}
    for model in MODELS:
        values = np.array([results[check][model]['rmse'] for check in ['DMC', 'EMC']])
        summary[model] = {'mean_rmse': float(values.mean()), 'std_rmse': float(values.std(ddof=1)),
                          'worst_rmse': float(values.max())}
    write_json(output / 'metrics.json', {'checks': results, 'chemical_summary': summary})
    write_csv(output / 'validation_predictions.csv', all_predictions)
    ranges['final_test'] = range_summary(train, test)
    write_json(diagnostics / 'feature_ranges.json', ranges)
    make_plots(diagnostics, all_predictions, {r['id']: r for r in train})
    prediction, final_fit = fit_predict(train, test, 'descriptor_gam')
    by_id = {r['id']: float(p) for r, p in zip(test, prediction)}
    submission = [{'id': r['id'], 'log_k': by_id[r['id']]} for r in sample]
    if len(submission) != len(test) or not np.isfinite([r['log_k'] for r in submission]).all():
        raise ValueError('Invalid final submission')
    provenance['fits']['final'] = final_fit
    write_json(output / 'run_manifest.json', provenance)
    write_report(output, results, summary, submission, final_fit)
    write_csv(output / 'submission.csv', submission, ['id', 'log_k'])
    print(f"Wrote {len(submission)} predictions to {output / 'submission.csv'}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/task'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/competition-gam-01'))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
