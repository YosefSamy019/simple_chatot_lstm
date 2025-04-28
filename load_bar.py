import time

_cache_currents = dict()
_cache_init_time = dict()

def LoadBarReset():
    _cache_currents.clear()
    _cache_init_time.clear()

def LoadBarWithAutoIncrement(key:str, total:int):
    current = _cache_currents.get(key, 1)
    init_time = _cache_init_time.get(key, time.time())

    percent = 100.0 * current / total
    percent = 100.0 if percent > 100.0 else percent
    percent = 0.0 if percent < 0.0 else percent

    timeTakenUntilNow = time.time() - init_time
    restTime = (timeTakenUntilNow / (percent + 0.1)) * (100.0 - percent)

    print(f'\rLoading: {percent:4.3f}%, Wait: {_formateSenconds(restTime)}',end='')

    _cache_currents[key] = _cache_currents.get(key, 1) + 1
    _cache_init_time[key] = _cache_init_time.get(key, time.time())


def _formateSenconds(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    seconds = seconds % 60

    hours = minutes //60
    minutes = minutes % 60

    return f"{hours:03d}:{minutes:02d}:{seconds:02d}"
