import pytest
from rmuc_student_distill import validate_resume_teacher

def test_teacher_change_requires_explicit_warm_start():
    previous = {'teacher_sha256': 'old'}
    with pytest.raises(ValueError, match='teacher changed'):
        validate_resume_teacher(previous, 'new')
    validate_resume_teacher(previous, 'new', True)
    assert previous['teacher_sha256'] == 'old'
    validate_resume_teacher(previous, 'old')
    validate_resume_teacher(None, 'new')
