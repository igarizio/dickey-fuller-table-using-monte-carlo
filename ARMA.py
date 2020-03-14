import numpy as np


class ARMA:
    """Class that generates WN, AR, MA and ARMA processes."""

    @staticmethod
    def generate_wn(n, sigma=1):
        """Generates a white noise series.

        The code follows:
        y_{t} = \epsilon_{t}

        Args:
            n: length of the series.
            sigma: standard deviation of the innovations.

        Returns:
            np.Array with the series.

        """
        return np.random.normal(0, sigma, size=n)

    @staticmethod
    def generate_ma(n, thetas, mu, sigma=1):
        """Generates a moving average series.

        The code follows:
        y_{t} = \mu + \epsilon_{t} + \theta_{1}\epsilon_{t-1} + \theta_{2}\epsilon_{t-2} + ... + \theta_{q}\epsilon_{t-q}

        Args:
            n: length of the series.
            thetas: list of thetas, in the order \theta_{1}, \theta_{2}, ..., \theta_{q}.
            mu: base constant.
            sigma: standard deviation of the innovations (optional).

        Returns:
            np.Array with the series.

        """
        q = len(thetas)
        adj_n = n + q  # We add q values because at the beginning we have no thetas available.
        e_series = ARMA.generate_wn(adj_n, sigma)  # Generating a white noise.

        ma = []
        for i in range(1, adj_n):
            visible_thetas = thetas[0:min(q, i)]  # At first, we only "see" some of the thetas.
            visible_e_series = e_series[i - min(q, i):i]  # The same happens to the white noise.

            reversed_thetas = visible_thetas[::-1]

            try:  # Getting e_t if we can.
                e_t = visible_e_series[-1]
            except IndexError:
                e_t = 0

            # Main equation.
            ma_t = mu + e_t + np.dot(reversed_thetas, visible_e_series)

            ma.append(ma_t)

        ma = ma[max(q-1, 0):]  # Dropping the first values that did not use all the thetas.

        return ma

    @staticmethod
    def generate_ar(n, phis, sigma=1):
        """Generates an autoregressive series.

        The code follows:
        y_{t} = \phi_{1} y_{t-1} + \phi_{2} y_{t-2} + ... + \phi_{p} y_{t-p} + \epsilon_{t}

        Args:
            n: length of the series.
            phis: list of thetas, in the order \phi_{1}, \phi_{2}, ..., \phi_{p}.
            sigma: standard deviation of the innovations (optional).

        Returns:
            np.Array with the series.

        """
        p = len(phis)
        adj_n = n + p  # We add q values because at the beginning we have no phis available.
        e_series = ARMA.generate_wn(adj_n, sigma)  # Generating a white noise.

        ar = [e_series[0]]  # We start the series with a random value
        for i in range(1, adj_n):
            visible_phis = phis[0:min(p, i)]  # At first, we only "see" some of the phis.
            visible_series = ar[i - min(p, i):i]  # The same happens to the white noise.

            reversed_phis = visible_phis[::-1]

            # Main equation.
            ar_t = e_series[i] + np.dot(reversed_phis, visible_series)

            ar.append(ar_t)

        ar = ar[p:]  # Dropping the first values that did not use all the phis.

        return ar

    @staticmethod
    def generate_arma(n, phis, thetas, mu, sigma=1):
        """Generates an autoregressive moving average series.

        The code follows:
        y_{t} = \mu + \phi_{1} y_{t-1} + \phi_{2} y_{t-2} + ... + \phi_{p} y_{t-p} + \epsilon_{t} + \theta_{1}\epsilon_{t-1} + \theta_{2}\epsilon_{t-2} + ... + \theta_{q}\epsilon_{t-q}

        Args:
            n: length of the series.
            phis: list of thetas, in the order \phi_{1}, \phi_{2}, ..., \phi_{p}.
            thetas: list of thetas, in the order \theta_{1}, \theta_{2}, ..., \theta_{q}.
            mu: base constant.
            sigma: standard deviation of the innovations (optional).

        Returns:
            np.Array with the series.

        """
        p = len(phis)
        q = len(thetas)

        adj_n = n + max(p, q)  # We use max to make sure we cover the lack of coefficients.
        e_series = ARMA.generate_wn(adj_n)  # Base white noise.

        arma = [e_series[0]]  # We start the series with a random value (same as AR).
        for i in range(1, adj_n):
            visible_phis = phis[0:min(p, i)]
            visible_thetas = thetas[0:min(q, i)]

            reversed_phis = visible_phis[::-1]
            reversed_thetas = visible_thetas[::-1]

            visible_series = arma[i - min(p, i):i]
            visible_e_series = e_series[i - min(q, i):i]
            
            try:  # Getting e_t if we can.
                e_t = visible_e_series[-1]
            except IndexError:
                e_t = 0

            # Main equation.
            ar_t = + np.dot(reversed_phis, visible_series)
            ma_t = mu + e_t + np.dot(reversed_thetas, visible_e_series)
            arma_t = ar_t + ma_t

            arma.append(arma_t)

        arma = arma[max(p, q):]  # Dropping the first values that did not use all the phis or thetas.

        return arma
