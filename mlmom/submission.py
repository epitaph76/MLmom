"""Validate organizer output completely before replacing a working submission."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .inputs import inside,read_table
from .pipeline_contracts import SubmissionArtifact,TaskContext,ExperimentRun,CandidateVersion
from .persistence import file_hash
from .evaluation import checked_predictions


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
