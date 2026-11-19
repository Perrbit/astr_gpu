    double trial[11],retreated[11],accepted_beta_value=0.,beta_lo=0.,beta_hi=1.;
    int projection;
    if(evaluate(1.,trial,projection)) {
        for(int c=0;c<11;++c)flux[c]=trial[c];accepted_beta=1.;return ok;
    }
    if(!evaluate(0.,best,projection))return projection==ok?200:100+projection;
    for(int iteration=0;iteration<ratio_iterations;++iteration) {
        const double beta=.5*(beta_lo+beta_hi),safe=beta*(1.-transport_gamma);
        if(evaluate(beta,trial,projection) && evaluate(safe,retreated,projection)) {
            beta_lo=beta;accepted_beta_value=safe;
            for(int c=0;c<11;++c)best[c]=retreated[c];
        } else beta_hi=beta;
    }
    for(int c=0;c<11;++c)flux[c]=best[c];accepted_beta=accepted_beta_value;return ok;
