"""Validate organizer output completely before replacing a working submission."""
from __future__ import annotations
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .inputs import inside,read_table
from .pipeline_contracts import SubmissionArtifact,TaskContext,ExperimentRun,CandidateVersion
from .persistence import atomic_json, file_hash
from .evaluation import checked_predictions


def validate_submission_file(context: TaskContext, test: pd.DataFrame, path: Path,
                             *, expected_hash: str | None = None) -> dict:
    """Check the serialized organizer CSV, including IDs and prediction values."""
    fmt = context.task.result_format
    digest = file_hash(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError('Submission checksum differs from the published artifact')
    dtypes = {column: test[column].dtype for column in context.task.id_columns}
    if context.submission_kind == 'label' and context.classes and all(isinstance(c, str) for c in context.classes):
        dtypes.update({column: str for column in fmt.prediction_columns})
    output = pd.read_csv(path, sep=fmt.delimiter or ',', encoding=fmt.encoding or 'utf-8', dtype=dtypes)
    if output.columns.tolist() != fmt.columns or len(output) != len(test):
        raise ValueError('Submission shape/columns differ from the task')
    if output.isna().any().any():
        raise ValueError('Submission contains missing values')
    ids = context.task.id_columns
    if ids and (output[ids].duplicated().any() or
                not output[ids].reset_index(drop=True).equals(test[ids].reset_index(drop=True))):
        raise ValueError('Submission IDs/order differ from test')
    values = output[fmt.prediction_columns].to_numpy()
    # A single binary-probability column is valid independently of class count.
    classes = context.classes if context.submission_kind == 'label' or values.shape[1] > 1 else None
    checked_predictions(values, len(test), context.submission_kind, classes=classes)
    return dict(status='passed', rows=len(output), columns=output.columns.tolist(), sha256=digest)


def snapshot_cycle_submission(context: TaskContext, test: pd.DataFrame, artifact: SubmissionArtifact,
                              run_dir: Path, round_id: str) -> dict:
    """Retain a verified answer for every cycle, also when the champion is unchanged."""
    if not round_id.startswith('r') or not round_id[1:].isdigit():
        raise ValueError('Invalid cycle ID')
    destination = run_dir / 'cycles' / round_id / 'submission.csv'
    report_path = destination.with_suffix('.json')
    if report_path.exists():
        import json
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['artifact'] != artifact.model_dump(mode='json'):
            raise ValueError('Attempt to replace a saved cycle answer')
        validate_submission_file(context, test, destination, expected_hash=artifact.sha256)
        return report
    source = Path(artifact.path)
    validate_submission_file(context, test, source, expected_hash=artifact.sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='submission.', suffix='.tmp', dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        verification = validate_submission_file(context, test, Path(temporary), expected_hash=artifact.sha256)
        if destination.exists() and file_hash(destination) != artifact.sha256:
            raise ValueError('Attempt to replace a saved cycle answer')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    report = dict(round_id=round_id, path=str(destination), artifact=artifact.model_dump(mode='json'),
                  verification=verification)
    atomic_json(report_path, report)
    return report


def publish_submission(context: TaskContext, test: pd.DataFrame, prediction, run: ExperimentRun,
                       candidate: CandidateVersion) -> SubmissionArtifact:
    fmt = context.task.result_format
    prediction = checked_predictions(prediction,len(test),context.submission_kind,classes=context.classes)
    if context.submission_kind=='probability' and prediction.ndim==2:
        if fmt.class_order:
            order=[list(map(str,context.classes)).index(str(label)) for label in fmt.class_order]
            prediction=prediction[:,order]
        elif prediction.shape[1]==2 and len(fmt.prediction_columns)==1:
            prediction=prediction[:,1]
    if prediction.ndim == 1: prediction = prediction[:,None]
    if prediction.shape[1] != len(fmt.prediction_columns):
        raise ValueError('Prediction width does not match required output columns')
    if context.task.id_columns:
        ids = test[context.task.id_columns]
        if ids.isna().any().any() or ids.duplicated().any(): raise ValueError('Test IDs are missing or duplicated')
        output = ids.copy()
    else:
        output = pd.DataFrame(index=np.arange(len(test)))
    for i,column in enumerate(fmt.prediction_columns): output[column] = prediction[:,i]
    if set(output.columns) != set(fmt.columns): raise ValueError('Submission schema differs from task')
    output = output.loc[:,fmt.columns]
    if context.files.sample_submission:
        sample = read_table(context.files.sample_submission)
        if sample.columns.tolist() != fmt.columns or len(sample)!=len(output):
            raise ValueError('Submission template disagrees with task schema/row count')
        if context.task.id_columns and not sample[context.task.id_columns].equals(test[context.task.id_columns].reset_index(drop=True)):
            raise ValueError('Template IDs/order disagree with test')
    destination = inside(Path(context.files.root),fmt.file_name or 'submission.csv')
    protected={Path(context.files.train),Path(context.files.test),Path(context.files.task),*[Path(p) for p in context.files.dictionaries]}
    if context.files.sample_submission:protected.add(Path(context.files.sample_submission))
    if context.files.manifest:protected.add(Path(context.files.manifest))
    if destination in protected:
        raise ValueError('Output path would overwrite task inputs')
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,temp = tempfile.mkstemp(prefix=destination.name+'.',suffix='.tmp',dir=destination.parent)
    os.close(fd)
    try:
        output.to_csv(temp,index=False,sep=fmt.delimiter or ',',encoding=fmt.encoding or 'utf-8')
        check = pd.read_csv(temp,sep=fmt.delimiter or ',',encoding=fmt.encoding or 'utf-8')
        if len(check)!=len(test) or check.columns.tolist()!=fmt.columns or check.isna().any().any():
            raise ValueError('Serialized submission failed verification')
        os.replace(temp,destination)
    finally:
        if Path(temp).exists(): Path(temp).unlink()
    return SubmissionArtifact(path=str(destination),experiment_id=run.experiment_id,candidate_id=candidate.candidate_id,
        model_role=run.role,rows=len(output),sha256=file_hash(destination))
