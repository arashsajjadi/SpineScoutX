# v1.8c — MedSAM2 spine prompt notes (Phase 4)

> Research-only · not diagnostic. For a controlled comparison with the v1.8b SAM2.1 baseline, real
> MedSAM2 used the **same center-box prompt** on the already-localized foraminal crops (the crops are
> centred on the localizer point, so a center box is the natural foramen prompt). A full prompt
> sweep (tight box / points / 3–5 slice propagation) was **deprioritized**: the morphometry-signal
> result (below) shows real MedSAM2's opening geometry is *flat* across severities and its severity
> signal is *weaker* than SAM2.1's already-redundant signal — prompt optimisation cannot make a
> redundant feature complementary to the image grader. Documented honestly rather than run to
> exhaustion on a feature proven non-complementary.
