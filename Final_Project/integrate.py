"""
integrate.py - reconciliation between the catalog and physical reality.
check_shelf is a multi-armed bandit: each shelf is an arm, reward is
whether the catalog matched ground truth, and once a shelf's reliability
crosses TRUST_THRESHOLD the physical check is skipped (exploit). compare_db
is a separate, non-Z3 confidence-threshold rule deciding whether to trust
a disagreeing check over the catalog.
"""

from pydantic import BaseModel, Field
import random
from typing import Optional
import digital_database
import shelves

class CheckSpec(BaseModel):
    book_id: str
    available: bool
    correct_prob: float = Field(ge=0.0, le=1.0) # the probability of the robot being correct
    skipped: bool = False # sees if the check is skipped due to high success rate of shelf
def compare_db(db_status: str, check: CheckSpec) -> dict:
    check_status = "available" if check.available else "checked_out"
    if check_status == db_status:
        return {"check_status": db_status, 
                "correct_prob": check.correct_prob,
                "difference": False,
                "content": "There is no difference between the digital database and the shelves"
                }
    else: 
        if check.correct_prob < 0.95:
            return {"check_status": None, 
                    "correct_prob": check.correct_prob,
                    "difference": True,
                    "content": f"There is a difference between the digital database, {db_status} and the shelves, {check_status}"
                    } 
        else: # this is only if the correct_prob is high enough to be trusted
            return {"check_status": check_status, 
                    "correct_prob": check.correct_prob,
                    "difference": True,
                    "content": f"There is a difference between the digital database, {db_status} and the shelves, {check_status} but the probability is high enough that we will trust the check over the database, and change the digital database to be {check_status}"
                    }

CHECK_RELIABILITY = 0.90 # ARBITRARY, CAN BE ADJUSTED AS NEEDED
HIGH_CONFIDENCE_LEVEL = (0.70, 0.99) # ARBITRARY, CAN BE ADJUSTED AS NEEDED
LOW_CONFIDENCE_LEVEL = (0.45, 0.70) # ARBITRARY, CAN BE ADJUSTED AS NEEDED
TRUST_THRESHOLD = 0.9 # ARBITRARY, CAN BE ADJUSTED AS NEEDED 
MIN_ATTEMPTS = 3 # ARBITRARY, CAN BE ADJUSTED AS NEEDED

def check_shelf(book_id: str, shelves_id: str,
                 digital_database_path: str = digital_database.DB_PATH,
                 shelves_path: str = shelves.DB_PATH,
                 seed: Optional[int] = None) -> CheckSpec:
    db_record = digital_database.get_book_record(book_id, db_path=digital_database_path)
    reliability = shelves.get_shelf_reliability(shelves_id, db_path=shelves_path)

    if reliability["attempts"] >= MIN_ATTEMPTS and reliability["success_score"] >= TRUST_THRESHOLD:
        trusted_available = db_record["status"] == "available" if db_record else False
        return CheckSpec(book_id=book_id, available=trusted_available,
                          correct_prob=reliability["success_score"], skipped=True)

    if seed is not None:
        random.seed(seed)

    truth = shelves.get_shelf_truth(book_id, db_path=shelves_path)
    if truth is None:
        return CheckSpec(book_id=book_id, available=False, correct_prob=0.5, skipped=False)

    true_present = bool(truth["physically_present"])
    if random.random() < CHECK_RELIABILITY:
        available = true_present
        correct_prob = random.uniform(*HIGH_CONFIDENCE_LEVEL)
    else:
        available = not true_present
        correct_prob = random.uniform(*LOW_CONFIDENCE_LEVEL)

    if db_record is not None:
        catalog_matches_reality = (db_record["status"] == "available") == true_present
        shelves.update_shelf_reliability(shelves_id, success=catalog_matches_reality, db_path=shelves_path)

    return CheckSpec(book_id=book_id, available=available, correct_prob=correct_prob, skipped=False)



