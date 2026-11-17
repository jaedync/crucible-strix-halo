"""A memory admission refusal must not reload the weights in a restart loop."""
import time


def test_admission_refusal_exits_4_without_restart(make_unit):
    unit = make_unit(env={'FAKE_ADMISSION_REFUSED': '1'})
    # start() waits for HTTP readiness, which a refused layout cannot provide.
    unit.start(wait=False)
    state, _ = unit.wait_for(lambda: (s := unit.state()).get('ActiveState') == 'failed' and s,
                             20, what='admission refusal')
    assert state['ExecMainStatus'] == '4', state
    assert state['NRestarts'] == '0', state
    time.sleep(2)  # Longer than the harness RestartSec (1 second).
    assert unit.state()['NRestarts'] == '0'
