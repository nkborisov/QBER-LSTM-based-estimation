namespace cpp chan_estimator_api

enum Code {
  OK = 1,
  TRY_LATER = 2,
  INTERNAL_ERROR = 3,
  INVALID_PARAMS = 4
}

struct Status {
    1: Code res,
    2: double est
}

service ChanEstimatorService {
    Status retrieveEst(
        1: double eMu
        2: double eMuX
        3: double eMuEma
        4: double eNu1
        5: double eNu2
        6: double qMu
        7: double qNu1
        8: double qNu2
        9: bool maintenance
    )
}
