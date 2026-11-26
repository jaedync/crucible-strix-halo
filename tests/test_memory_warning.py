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


def test_memory_checks_keep_their_fail_level_and_name_the_holders(tmp_path):
    # The levels stay. The fix text gives the holders and advice.
    # A 64 GB machine (lowmem): below the 96 GB class the memory check fails with the reason (tools/crucible_cli/machine.py);
    # a 96 GB machine passes it with the 3.05 pack (tests/test_crucible_machine.py).
    host = wire(tmp_path, mem_total_kb=64000000, gtt_used=31000000000)
    process(host, 4120, 'gpu-holder', 1000000, ['drm-client-id:\t7\ndrm-memory-gtt:\t31000000 KiB\n'])
    s = settings.build(make_repo(tmp_path), host)
    checks = {r.id: r for r in doctor.run_checks(host, s)}
    for key in ('memory', 'gpu-free'):
        assert checks[key].status == 'fail'
        assert checks[key].blocks == ('start',)
    assert 'Crucible needs a 96 GB or 128 GB machine.' in checks['memory'].fix
    assert 'Crucible does not stop other programs.' in checks['gpu-free'].fix
    assert 'PID 4120 gpu-holder' in checks['gpu-free'].fix
    assert checks['memory-available'].status == 'pass'
    (host.proc / 'meminfo').write_text('MemTotal: 64000000 kB\nMemAvailable: 1000000 kB\n')
    checks = {r.id: r for r in doctor.run_checks(host, s)}
    assert checks['memory-available'].status == 'warn'
    assert 'Crucible does not stop other programs.' in checks['memory-available'].fix


def test_other_users_and_unreadable_files_are_skipped(tmp_path):
    host = wire(tmp_path)
    process(host, 4120, 'other', 1000, ['not a DRM file'])
    host.uid += 1
    assert memory_holders(host) == []


def test_low_available_memory_is_no_warning_while_our_own_server_holds_it(tmp_path):
    # The server itself holds most of the memory while it runs: a doctor run then must not warn about it.
    host = wire(tmp_path)
    repo = make_repo(tmp_path)
    (host.proc / 'meminfo').write_text('MemTotal: 130000000 kB\nMemAvailable: 1000000 kB\n')
    (repo / '.crucible').mkdir()
    (repo / '.crucible/server.pid').write_text('40001 1\n')
    host.runner.live.add(40001)
    checks = {r.id: r for r in doctor.run_checks(host, settings.build(repo, host))}
    assert checks['memory-available'].status == 'pass', checks['memory-available']
    assert 'Crucible server' in checks['memory-available'].detail


def test_free_cma_memory_names_kho_off(tmp_path):
    # Issue 18: Ubuntu 26.04 (kernel 7.0) keeps the Kexec HandOver scratch as CMA. MemAvailable counts it, but GPU
    # memory (GTT) cannot use it, so the engine fit and its guards have less room than MemAvailable says.
    host = wire(tmp_path)
    s = settings.build(make_repo(tmp_path), host)
    base = 'MemTotal: 127098440 kB\nMemAvailable: 100000000 kB\n'
    (host.proc / 'meminfo').write_text(base + 'CmaTotal: 0 kB\nCmaFree: 13771364 kB\n')  # the reporter's host
    checks = {r.id: r for r in doctor.run_checks(host, s)}
    cma = checks['memory-cma']
    assert cma.status == 'warn', cma
    assert '13.1 GiB' in cma.detail
    assert 'kho=off' in cma.fix and 'update-grub' in cma.fix
    # Each boot loader gets its own steps here: the GTT check passes on this host and shows no fix text (review LOW 1).
    for step in ('/boot/loader/entries/', '/etc/kernel/cmdline', 'mkinitcpio -P', 'another boot loader: see its documentation'):
        assert step in cma.fix, step
    assert 'GTT check' not in cma.fix
    # Two facts with their condition (review LOW 3): another cma= parameter keeps CMA, and the smaller layout needs --fit auto.
    assert 'cma=' in cma.fix and '/proc/cmdline' in cma.fix
    assert '--fit auto' in cma.fix and '--fit off' in cma.fix
    assert checks['memory-available'].detail.startswith('82.2 GiB'), checks['memory-available']
    # 1 GiB or less of free CMA (the test machine: 0 kB, or no CmaFree line): no check line, the other checks as before.
    for extra, available in (('CmaTotal: 0 kB\nCmaFree: 0 kB\n', '95.4 GiB'), ('CmaFree: 1048576 kB\n', '94.4 GiB'),
                             ('', '95.4 GiB')):
        (host.proc / 'meminfo').write_text(base + extra)
        checks = {r.id: r for r in doctor.run_checks(host, s)}
        assert 'memory-cma' not in checks
        assert checks['memory-available'].detail.startswith(available), checks['memory-available']
