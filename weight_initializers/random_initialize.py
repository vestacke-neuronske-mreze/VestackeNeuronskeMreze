from backend.backend import xp

xp.random.seed(0)


def rand_init(n: int, m: int, init_mode: str = 'xavier_uniform') -> xp.ndarray:

    if init_mode.lower() == 'he_uniform':
        a = xp.sqrt(1 / m)
        return xp.random.uniform(size=(n, m), low=-a, high=a)
    if init_mode == 'xavier_uniform':
        limit = xp.sqrt(6 / (n+m))
        return xp.random.uniform(size=(n, m), low=-limit, high=limit)
    if init_mode.lower() == 'xavier_normal':
        std = xp.sqrt(2 / (m + n))
        return xp.random.normal(loc=0, scale=std, size=(n, m))

    return xp.random.randn(n, m)


