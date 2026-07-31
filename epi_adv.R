suppressMessages({library(survey); library(splines); library(jsonlite)})
options(survey.lonely.psu="adjust")
cfg<-fromJSON("epi_config.json"); d<-read.csv("epi_analytic.csv")
y<-cfg$outcome; cov<-cfg$cov; xterm<-cfg$exposure; xbin<-cfg$exposure_bin; xcont<-cfg$exposure_cont
des<-svydesign(ids=~psu,strata=~kstrata,weights=~wt_pool,data=d,nest=TRUE)

## VIF (based on unweighted design matrix)
terms<-c(xterm,cov); vif<-c()
for(t in terms){
  ot<-setdiff(terms,t)
  r2<-tryCatch(summary(lm(as.formula(paste(t,"~",paste(ot,collapse="+"))),data=d))$r.squared,error=function(e)NA)
  vif<-c(vif, if(is.na(r2)) NA else round(1/(1-r2),2))
}
write.csv(data.frame(term=terms,VIF=vif),"epi_vif.csv",row.names=FALSE)

## RCS nonlinearity (continuous exposure only)
pnl<-NA
if(!is.null(xcont) && !is.na(xcont) && nzchar(xcont)){
  ff<-tryCatch(svyglm(as.formula(paste(y,"~ ns(",xcont,",3)+",paste(cov,collapse="+"))),design=des,family=quasibinomial()),error=function(e)NULL)
  fl<-tryCatch(svyglm(as.formula(paste(y,"~",xcont,"+",paste(cov,collapse="+"))),design=des,family=quasibinomial()),error=function(e)NULL)
  if(!is.null(ff)&&!is.null(fl)) pnl<-tryCatch(as.numeric(anova(fl,ff)$p)[1],error=function(e)NA)
}
write.csv(data.frame(test="RCS_nonlinearity_p",value=pnl),"epi_rcs.csv",row.names=FALSE)

## PS, IPTW, Love, causal estimation (binary exposure)
dd<-d[!is.na(d[[xbin]]) & complete.cases(d[,cov,drop=FALSE]) & !is.na(d[[y]]),]
desb<-svydesign(ids=~psu,strata=~kstrata,weights=~wt_pool,data=dd,nest=TRUE)
psf<-svyglm(as.formula(paste(xbin,"~",paste(cov,collapse="+"))),design=desb,family=quasibinomial())
ps<-as.numeric(predict(psf,type="response"))
ptx<-as.numeric(coef(svymean(as.formula(paste0("~",xbin)),desb))[1])
A<-dd[[xbin]]; sw<-weights(desb,"sampling")
iptw<-ifelse(A==1, ptx/ps, (1-ptx)/(1-ps)); cw<-sw*iptw          # stabilized IPTW x sampling weight
# Love plot SMD (before=sw, after=cw)
wm<-function(x,w) sum(w*x)/sum(w); wv<-function(x,w){m<-wm(x,w);sum(w*(x-m)^2)/sum(w)}
smd1<-function(x,A,w){
  if(length(unique(x[!is.na(x)]))<=2){p1<-wm(x[A==1],w[A==1]);p0<-wm(x[A==0],w[A==0]);pb<-(p1+p0)/2
    if(pb<=0||pb>=1) return(NA); (p1-p0)/sqrt(pb*(1-pb))}
  else{m1<-wm(x[A==1],w[A==1]);m0<-wm(x[A==0],w[A==0]);s<-sqrt((wv(x[A==1],w[A==1])+wv(x[A==0],w[A==0]))/2)
    if(s==0) return(NA); (m1-m0)/s}
}
love<-data.frame()
for(c0 in cov){x<-dd[[c0]]; ok<-!is.na(x)
  love<-rbind(love,data.frame(covariate=c0,before=round(abs(smd1(x[ok],A[ok],sw[ok])),3),after=round(abs(smd1(x[ok],A[ok],cw[ok])),3)))}
write.csv(love,"epi_love.csv",row.names=FALSE)

orci<-function(fit,term){s<-summary(fit)$coefficients; b<-s[term,1];se<-s[term,2];c(exp(b),exp(b-1.96*se),exp(b+1.96*se))}
res<-data.frame()
addm<-function(name,est,lo,hi,scale) res<<-rbind(res,data.frame(method=name,estimate=round(est,3),lo=round(lo,3),hi=round(hi,3),scale=scale))
# Crude / Min-adj(age+sex) / Full-adj  (OR)
fc0<-svyglm(as.formula(paste(y,"~",xbin)),design=desb,family=quasibinomial()); r<-orci(fc0,xbin); addm("Crude",r[1],r[2],r[3],"OR")
mincov<-intersect(c("age","men"),cov)
if(length(mincov)>0){fm<-svyglm(as.formula(paste(y,"~",xbin,"+",paste(mincov,collapse="+"))),design=desb,family=quasibinomial());r<-orci(fm,xbin);addm("Min-adj (age,sex)",r[1],r[2],r[3],"OR")}
ff<-svyglm(as.formula(paste(y,"~",xbin,"+",paste(cov,collapse="+"))),design=desb,family=quasibinomial()); r<-orci(ff,xbin); addm("Full-adj",r[1],r[2],r[3],"OR")
# IPTW (OR, combined weight)
desi<-svydesign(ids=~psu,strata=~kstrata,weights=~cw,data=cbind(dd,cw=cw),nest=TRUE)
fi<-svyglm(as.formula(paste(y,"~",xbin)),design=desi,family=quasibinomial()); r<-orci(fi,xbin); addm("IPTW",r[1],r[2],r[3],"OR")
# G-computation (risk difference RD)
gm<-svyglm(as.formula(paste(y,"~",xbin,"+",paste(cov,collapse="+"))),design=desb,family=quasibinomial())
d1<-dd; d1[[xbin]]<-1; d0<-dd; d0[[xbin]]<-0
m1<-as.numeric(predict(gm,newdata=d1,type="response")); m0<-as.numeric(predict(gm,newdata=d0,type="response"))
rd_g<-wm(m1,sw)-wm(m0,sw)
# AIPW (doubly robust RD)
aipw_i<-(A*(dd[[y]]-m1)/ps + m1) - ((1-A)*(dd[[y]]-m0)/(1-ps) + m0)
rd_a<-wm(aipw_i,sw); se_a<-sqrt(sum((sw/sum(sw))^2*(aipw_i-rd_a)^2))
addm("G-computation",rd_g,NA,NA,"RD"); addm("AIPW",rd_a,rd_a-1.96*se_a,rd_a+1.96*se_a,"RD")
write.csv(res,"epi_table6.csv",row.names=FALSE)
cat("epi_adv.R OK\n")
