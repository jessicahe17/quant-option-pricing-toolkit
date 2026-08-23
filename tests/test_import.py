def test_import():
    import option_pricing


def test_public_subpackage_imports():
    from option_pricing.instruments import (
        Equity,
        ExerciseStyle,
        Option,
        OptionType,
        Underlying,
    )
    from option_pricing.market import MarketData, MarketEnvironment
    from option_pricing.models import (
        BlackScholesModel,
        Greeks,
        PricingModel,
        PricingResult,
    )
    from option_pricing.payoffs import (
        AsianCall,
        EuropeanCall,
        EuropeanPut,
        PathDependentPayoff,
        Payoff,
        PayoffRequirement,
        TerminalPayoff,
    )
    from option_pricing.simulation import (
        AntitheticSimulationResult,
        AntitheticSimulator,
        GBM,
        MonteCarloPricer,
        MonteCarloPricingResult,
        SimulationResult,
        Simulator,
        StochasticProcess,
    )
    from option_pricing.validation import (
        GBMDistributionValidationResult,
        GBMMomentValidationResult,
        theoretical_log_mean,
        theoretical_log_std,
        theoretical_mean,
        theoretical_variance,
        validate_gbm_distribution,
        validate_gbm_moments,
    )