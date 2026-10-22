from typing import Any, Tuple

def get_laterality(ds: Any) -> Tuple[str, str]:
    tag = getattr(ds, "ImageLaterality", None)
    if tag is not None and str(tag).strip():
        val = str(tag).strip().upper()
        if val in ("L", "LEFT"):
            return "L", "ImageLaterality"
        if val in ("R", "RIGHT"):
            return "R", "ImageLaterality"

    body_part = getattr(ds, "BodyPartExamined", None)
    if body_part is not None:
        bp_str = str(body_part).strip().upper()
        if bp_str.startswith("L") or "LEFT" in bp_str:
            return "L", "BodyPartExamined"
        if bp_str.startswith("R") or "RIGHT" in bp_str:
            return "R", "BodyPartExamined"

    series_desc = getattr(ds, "SeriesDescription", None)
    if series_desc is not None:
        sd_str = str(series_desc).strip().upper()
        tokens = sd_str.replace("_", " ").replace("-", " ").split()
        if "RT" in tokens or "RIGHT" in tokens or "R" in tokens:
            return "R", "SeriesDescription"
        if "LT" in tokens or "LEFT" in tokens or "L" in tokens:
            return "L", "SeriesDescription"

    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is not None and len(iop) >= 6:
        try:
            x_vec = [float(v) for v in iop[:3]]
            if x_vec[0] > 0.5:
                return "R", "ImageOrientationPatient"
            if x_vec[0] < -0.5:
                return "L", "ImageOrientationPatient"
        except (ValueError, TypeError):
            pass

    return "Unknown", "none"

if __name__ == "__main__":
    class DummyDS:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    ds_lat_l = DummyDS(ImageLaterality="L")
    ds_lat_r = DummyDS(ImageLaterality="R")
    ds_iop_l = DummyDS(ImageOrientationPatient=[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    ds_iop_r = DummyDS(ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    ds_sd_r = DummyDS(SeriesDescription="Knee RT Coronal")
    ds_sd_l = DummyDS(SeriesDescription="Knee LT Sagittal")
    ds_unknown = DummyDS()

    assert get_laterality(ds_lat_l) == ("L", "ImageLaterality")
    assert get_laterality(ds_lat_r) == ("R", "ImageLaterality")
    assert get_laterality(ds_iop_l) == ("L", "ImageOrientationPatient")
    assert get_laterality(ds_iop_r) == ("R", "ImageOrientationPatient")
    assert get_laterality(ds_sd_r) == ("R", "SeriesDescription")
    assert get_laterality(ds_sd_l) == ("L", "SeriesDescription")
    assert get_laterality(ds_unknown) == ("Unknown", "none")
    print("Laterality self-test passed successfully.")
