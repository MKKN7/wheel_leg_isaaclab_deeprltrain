"""Explicit critic warmup rejection, not a behavior or convergence certificate."""
def check_warmup(history, max_loss, max_bootstrap):
    if not history:raise ValueError('No completed critic warmup')
    recent=history[-min(10,len(history)):]
    import math,statistics
    losses=[float(x['critic_loss']) for x in recent]
    tails=[abs(float(x['bootstrap_return'])) for x in recent]
    finite=all(math.isfinite(x) for x in losses+tails)
    report={'finite':finite,'median_critic_loss':statistics.median(losses),'median_abs_bootstrap':statistics.median(tails),'max_loss':max_loss,'max_bootstrap':max_bootstrap,'samples':len(recent)}
    report['passed']=finite and report['median_critic_loss']<=max_loss and report['median_abs_bootstrap']<=max_bootstrap
    return report
