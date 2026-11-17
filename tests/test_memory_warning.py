from crucible_cli_fakes import wire, make_repo
from crucible_cli import doctor, settings
from crucible_cli.memfit import memory_holders, memory_advice


def process(host, pid, name, rss, fdinfo):
    entry = host.proc / str(pid)
    (entry / 'fdinfo').mkdir(parents=True)
    (entry / 'comm').write_text(name + '\n')
    (entry / 'status').write_text(f'VmRSS:\t{rss} kB\n')
    for fd, text in enumerate(fdinfo):
        (entry / 'fdinfo' / str(fd)).write_text(text)


def test_holders_show_rss_and_deduplicate_drm_clients(tmp_path):
    host = wire(tmp_path)
    drm = 'drm-client-id:\t7\ndrm-memory-gtt:\t31000000 KiB\ndrm-total-gtt:\t31000000 KiB\n'
    process(host, 4120, 'ollama', 1000000, [drm, drm])
    process(host, 2210, 'firefox', 6000000, [])
    holders = memory_holders(host)
    assert [(p.pid, p.name, p.rss, p.gtt) for p in holders] == [
        (4120, 'ollama', 1024000000, 31744000000), (2210, 'firefox', 6144000000, 0)]
    text = memory_advice(host)
    assert 'PID 4120 ollama: 1.0 GB host, 31.7 GB GPU' in text
    assert 'PID 2210 firefox: 6.1 GB host, 0.0 GB GPU' in text
    assert 'Only processes owned by your user are listed.' in text
    assert 'Crucible does not stop other programs.' in text
    assert '3.05 pack' in text and '--ctx' in text and '--conversation-cache-mib' in text and '--streams' in text
    assert host.runner.terminated == []


def test_memory_checks_warn_and_do_not_block_start(tmp_path):
    host = wire(tmp_path, mem_total_kb=96000000, gtt_used=31000000000)
    s = settings.build(make_repo(tmp_path), host)
    checks = {r.id: r for r in doctor.run_checks(host, s)}
    for key in ('memory', 'gpu-free'):
        assert checks[key].status == 'warn'
        assert 'start' not in checks[key].blocks
    assert checks['memory-available'].status == 'pass'
    (host.proc / 'meminfo').write_text('MemTotal: 96000000 kB\nMemAvailable: 1000000 kB\n')
    checks = {r.id: r for r in doctor.run_checks(host, s)}
    assert checks['memory-available'].status == 'warn'
    assert 'Crucible does not stop other programs.' in checks['memory-available'].fix


def test_other_users_and_unreadable_files_are_skipped(tmp_path):
    host = wire(tmp_path)
    process(host, 4120, 'other', 1000, ['not a DRM file'])
    host.uid += 1
    assert memory_holders(host) == []
