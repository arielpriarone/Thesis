# Ariel Priarone - Master Degree thesys
code repository for thesys work.

Configuration file as the following:
    {
    "Database": {                                       __Database section__   
        "URI": "mongodb://localhost:27017",             *URI to the MondoDB connection*
        "db": "Shaft",                                  *Name of the database*
        "collection":{                                  __Collecitons in the database__
            "back"          : "BACKUP",                 *Name of the backup raw data colleciton*
            "raw"           : "RAW",                    *Name of the raw data colleciton (to be depleted by FA)*
            "unconsumed"    : "UNCONSUMED",             *Name of the unconsumed features colleciton*
            "healthy"       : "HEALTHY",                *Name of the healty features colleciton*
            "quarantined"   : "QUARANTINED",            *Name of the quarantined features colleciton*
            "faulty"        : "FAULTY",                 *Name of the faulty features colleciton*
            "models"        : "MODELS"                  *Name of the models instance colleciton*
        }
    },
    "Features": ["WavPowers, Mean, Std"],               __Features to be extracted__
    "wavelet": {                                        __Wavelet section__
        "type"              : "db10",                   *Wavelet type*
        "mode"              : "symmetric",              *Wavelet mode*
        "maxlevel"          : 6                         *Wavelet max level*
    }
}
