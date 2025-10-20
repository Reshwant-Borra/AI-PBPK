from typing import Literal, Optional
from pydantic import BaseModel


OrganName = Literal[
	"plasma", "tumor", "liver", "spleen", "lung", "heart", "kidney", "rest_of_body", "extracellular"
]


class CanonicalRow(BaseModel):
	study_id: str
	nanoparticle_id: str
	core_material: Optional[str] = None
	shape: Optional[str] = None
	hydrodynamic_diameter_nm: Optional[float] = None
	zeta_potential_mV: Optional[float] = None
	surface_coating: Optional[str] = None
	tumor_model: Optional[str] = None
	time_h: float
	organ: OrganName
	conc_value: float
	conc_unit: str
	conc_mg_per_kg: Optional[float] = None



