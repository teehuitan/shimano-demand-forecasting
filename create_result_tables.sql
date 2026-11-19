-- ############################################################################################
-- ## DESTRUCTIVE. This script DROPS every table below, PredictionResultLog included - which is
-- ## the append-only history of every forecast ever produced and cannot be regenerated.
-- ## Run this ONLY to build a brand-new empty database.
-- ############################################################################################

-- Drops every table this schema has ever had, named explicitly - the 6 the current design
-- defines below, plus RunLog .
DROP TABLE IF EXISTS DemandForecast.PredictionResult;
DROP TABLE IF EXISTS DemandForecast.PredictionResultLog;
DROP TABLE IF EXISTS DemandForecast.ActualOrder;
DROP TABLE IF EXISTS DemandForecast.ProductLifecycle;
DROP TABLE IF EXISTS DemandForecast.ProductSuccessor;
DROP TABLE IF EXISTS DemandForecast.OutstandingOrder;
DROP TABLE IF EXISTS DemandForecast.RunLog;

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'DemandForecast')
    EXEC('CREATE SCHEMA DemandForecast');
GO


-- Holds only the current settled month's results (see PredictionResultLog for history): each day's
-- run adds its rows, a rerun replaces its own OriginDate. Wiped (with OutstandingOrder) by the run
-- that first writes the new month's ActualOrder, in one transaction. Filter on OriginDate for one day.
--
-- OriginDate is the cutoff date: the last completed day of the order book this forecast saw. A
-- month-end OriginDate is day 0 of that month's cycle (the month just settled); any other date is
-- that calendar day, with the month before as settled.
CREATE TABLE DemandForecast.PredictionResult (
    Id                BIGINT IDENTITY(1,1) NOT NULL,
    Plant             VARCHAR(50)   NOT NULL,
    PartCode          VARCHAR(50)   NOT NULL,
    PlanModelCode     VARCHAR(50)   NOT NULL,   -- forecast grain; SalesModelCode/PlanModelText via ProductLifecycle
    OriginDate        DATETIME2     NOT NULL,   -- cutoff date of the order book - see table comment
    TargetYearMonth   VARCHAR(10)   NOT NULL,
    Horizon           INT           NOT NULL,
    KnownGrossOrder   FLOAT         NOT NULL,
    KnownNetOrder     FLOAT         NOT NULL,   -- KnownGrossOrder minus known-canceled orders
    Method            VARCHAR(200)  NOT NULL,   -- e.g. 'model_v1'
    PredictedGrossOrder FLOAT       NOT NULL,
    PredictedNetOrder FLOAT         NOT NULL,   -- PredictedGrossOrder minus known-canceled orders
    InsertedAt        DATETIME2     NOT NULL DEFAULT SYSDATETIME(),
    CONSTRAINT PK_PredictionResult_Id PRIMARY KEY (Id)
);
CREATE INDEX IX_PredictionResult_Plant_PlanModelCode_TargetYearMonth_InsertedAt
    ON DemandForecast.PredictionResult (Plant, PlanModelCode, TargetYearMonth, InsertedAt);


-- PredictionResult's shape plus SalesModelCode (frozen as of the prediction), append-only - every
-- historical prediction ever produced, never overwritten or pruned. Write-only: the pipeline never
-- reads it back, so it has no index beyond the primary key. A rerun at the same OriginDate (a manual
-- rerun, or a retry after a failed OutstandingOrder step) appends all its rows again, so the latest
-- row per forecast and OriginDate, by Id, is always the one PredictionResult holds.
CREATE TABLE DemandForecast.PredictionResultLog (
    Id                BIGINT IDENTITY(1,1) NOT NULL,
    Plant             VARCHAR(50)   NOT NULL,
    PartCode          VARCHAR(50)   NOT NULL,
    SalesModelCode    VARCHAR(50)   NOT NULL,
    PlanModelCode     VARCHAR(50)   NOT NULL,
    OriginDate        DATETIME2     NOT NULL,   -- cutoff date of the order book
    TargetYearMonth   VARCHAR(10)   NOT NULL,
    Horizon           INT           NOT NULL,
    KnownGrossOrder   FLOAT         NOT NULL,
    KnownNetOrder     FLOAT         NOT NULL,   -- KnownGrossOrder minus known-canceled orders
    Method            VARCHAR(200)  NOT NULL,
    PredictedGrossOrder FLOAT       NOT NULL,
    PredictedNetOrder FLOAT         NOT NULL,   -- PredictedGrossOrder minus known-canceled orders
    InsertedAt        DATETIME2     NOT NULL DEFAULT SYSDATETIME(),
    CONSTRAINT PK_PredictionResultLog_Id PRIMARY KEY (Id)
);


CREATE TABLE DemandForecast.ActualOrder (
    Id                BIGINT IDENTITY(1,1) NOT NULL,
    Plant             VARCHAR(50)   NOT NULL,
    PartCode          VARCHAR(50)   NOT NULL,
    PlanModelCode     VARCHAR(50)   NOT NULL,   -- SalesModelCode/PlanModelText via ProductLifecycle
    TargetYearMonth   VARCHAR(10)   NOT NULL,
    ActualGrossOrder  FLOAT         NOT NULL,
    ActualNetOrder    FLOAT         NOT NULL,   -- ActualGrossOrder minus orders with a canceled status
    SalesResult       FLOAT         NULL,       -- HistoricalSales_LocalAndExport qty; NULL = outside its loaded Period range
    InsertedAt        DATETIME2     NOT NULL DEFAULT SYSDATETIME(),   -- set once, at first insert
    UpdatedAt         DATETIME2     NULL,                             -- set whenever an existing row's value changes
    CONSTRAINT PK_ActualOrder_Id PRIMARY KEY (Id),
    CONSTRAINT UQ_ActualOrder_Plant_PlanModelCode_TargetYearMonth UNIQUE (Plant, PlanModelCode, TargetYearMonth)   -- PartCode/SalesModelCode omitted: derivable 1:1 from PlanModelCode
);


-- Upserted, never wiped-and-reinserted
CREATE TABLE DemandForecast.ProductLifecycle (
    Id                BIGINT IDENTITY(1,1) NOT NULL,
    Plant             VARCHAR(50)   NOT NULL,
    PartCode          VARCHAR(50)   NOT NULL,
    SalesModelCode    VARCHAR(50)   NOT NULL,
    PlanModelCode     VARCHAR(50)   NOT NULL,
    PlanModelText     NVARCHAR(40)  NULL,       -- TFGMaterial ProductAbbreviation, display only; NULL = none in TFG
    StartYearMonth    VARCHAR(10)   NULL,       -- NULL = unresolved (no TFGMaterial ProductStartDate, or Start>End nulled)
    EndYearMonth      VARCHAR(10)   NULL,       -- NULL = unresolved / still active
    InsertedAt        DATETIME2     NOT NULL DEFAULT SYSDATETIME(),   -- set once, at first insert
    UpdatedAt         DATETIME2     NULL,                             -- set whenever an existing row's value changes
    CONSTRAINT PK_ProductLifecycle_Id PRIMARY KEY (Id),
    CONSTRAINT UQ_ProductLifecycle_Plant_PlanModelCode UNIQUE (Plant, PlanModelCode)   -- PartCode/SalesModelCode omitted: derivable 1:1 from PlanModelCode
);


-- Holds the current month's daily snapshots, one per source file (a rerun replaces its own file).
-- Wiped with PredictionResult at the new month (see above). Filter on SourceFileName for one day. No log table - display-only, not used in training. No OriginDate here - unlike PredictionResult, there's no "last completed day"
-- concept; instead SourceFileName records exactly which daily file this was computed from, and
-- TargetYearMonth records N+1 (the month this outstanding figure is displayed alongside).
CREATE TABLE DemandForecast.OutstandingOrder (
    Id                BIGINT IDENTITY(1,1) NOT NULL,
    Plant             VARCHAR(50)   NOT NULL,
    PartCode          VARCHAR(50)   NOT NULL,
    PlanModelCode     VARCHAR(50)   NOT NULL,   -- SalesModelCode/PlanModelText via ProductLifecycle
    SourceFileName    VARCHAR(200)  NOT NULL,   -- e.g. '20260914.xlsx'
    TargetYearMonth   VARCHAR(10)   NOT NULL,   -- N+1
    OutstandingOrder  FLOAT         NOT NULL,
    InsertedAt        DATETIME2     NOT NULL DEFAULT SYSDATETIME(),
    CONSTRAINT PK_OutstandingOrder_Id PRIMARY KEY (Id)
);
CREATE INDEX IX_OutstandingOrder_Plant_PlanModelCode_TargetYearMonth
    ON DemandForecast.OutstandingOrder (Plant, PlanModelCode, TargetYearMonth);


-- One row per (old code, new code it was replaced by) - a many-to-many self-relationship (a code
-- can have multiple successors and/or predecessors), so the reverse direction (does X have a
-- predecessor) is answered via WHERE SuccessorCode = X, not a second stored list. Rows are
-- added/updated/deleted individually as needed, not a per-run refresh. SourcePlant is provenance
-- only (which plant provides this information).
CREATE TABLE DemandForecast.ProductSuccessor (
    Id                BIGINT IDENTITY(1,1) NOT NULL,
    SalesModelCode    VARCHAR(50)   NOT NULL,   -- old code
    SuccessorCode     VARCHAR(50)   NOT NULL,   -- new code it was replaced by
    SourcePlant       VARCHAR(50)   NOT NULL,   -- which plant provides this information
    InsertedAt        DATETIME2     NOT NULL DEFAULT SYSDATETIME(),   -- set once, at first insert
    UpdatedAt         DATETIME2     NULL,                             -- set whenever an existing row's value changes
    CONSTRAINT PK_ProductSuccessor_Id PRIMARY KEY (Id),
    CONSTRAINT UQ_ProductSuccessor_SalesModelCode_SuccessorCode UNIQUE (SalesModelCode, SuccessorCode)
);
CREATE INDEX IX_ProductSuccessor_SuccessorCode ON DemandForecast.ProductSuccessor (SuccessorCode);
