"""Persistent EDA coverage metadata; never an experiment admission decision."""
import copy


EDA_TOPICS = {
    'target': 'Train-only label balance or quantiles/tails; task-specific targets and weights.',
    'missingness': 'Column/row missing rates, joint patterns, target/group/source dependence.',
    'feature_distributions': 'Numeric ranges/quantiles and categorical frequencies with reviewed-column scope.',
    'duplicates': 'Exact/feature-only duplicates, conflicting labels and repeated events versus copied rows.',
    'cardinality': 'Constants, rare levels, near-unique fields, vocabulary sizes and unseen levels.',
    'train_inference_shift': 'Comparable feature coverage/distributions/schema; never inference labels.',
    'group_structure': 'Entity/group support, imbalance, repeated observations and split implications.',
    'temporal_patterns': 'Order, seasonality, drift, gaps, cutoff and availability; explain absent time inputs.',
    'anomalies': 'Invalid units/ranges, infinities, impossible combinations and rare valid cases.',
    'leakage': 'Potential post-outcome fields, target proxies, split overlap and availability explanations.',
}
EDA_STATUSES = ('pending','screened','effect_checked','not_applicable')


def eda_matrix(previous=None, update=None):
    """Normalize coverage declarations, without judging their scientific correctness."""
    result={topic:dict(status='pending') for topic in EDA_TOPICS}
    for values in (previous,update):
        if not isinstance(values,dict):continue
        for topic,entry in values.items():
            if topic not in result or not isinstance(entry,dict):continue
            entry=copy.deepcopy(entry)
            if entry.get('status') not in EDA_STATUSES:
                entry['status']='pending'
            if entry['status']=='not_applicable' and not entry.get('reason'):
                entry['status']='pending'
            if entry['status'] in {'screened','effect_checked'} and (not entry.get('scope') or not entry.get('summary')):
                entry['status']='pending'
            result[topic]=entry
    return result


def pending_eda(matrix):
    return [f'EDA screening pending: {topic}' for topic,entry in matrix.items()
            if entry.get('status')=='pending']


def eda_plan(matrix, *, train_hash):
    return dict(scope='Registered current train; inference features only for shift. Profile scope may differ.',
        train_hash=train_hash,statuses=list(EDA_STATUSES),
        topics={topic:dict(required_screen=description,progress=matrix[topic])
                for topic,description in EDA_TOPICS.items()},
        output='agent_notes.eda_matrix: topic -> status, scope, reviewed columns/support, evidence, alternatives, next check',
        effect_contract='Observation.rows is subgroup support; numbers include population denominator/effect size; agent_notes.alternative_checks records performed controls or pending work.')
