import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  Zap,
  RefreshCw,
  BarChart3,
  PieChart as PieChartIcon,
  Settings,
  ChevronRight,
  Play,
  Pause,
  AlertCircle,
  CheckCircle2,
  Clock,
  Sparkles,
} from "lucide-react";

// API Configuration - uses env var in production, proxy in development
const API_BASE = import.meta.env.VITE_API_URL 
  ? `${import.meta.env.VITE_API_URL}/api` 
  : "/api";

// Predefined baskets
const BASKET_OPTIONS = {
  tech_leaders: {
    name: "Tech Leaders",
    tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
  },
  finance: {
    name: "Finance",
    tickers: ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP"],
  },
  healthcare: {
    name: "Healthcare",
    tickers: ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO"],
  },
  diversified: {
    name: "Diversified",
    tickers: ["AAPL", "JPM", "JNJ", "XOM", "WMT", "GOOGL", "PG", "UNH"],
  },
};

// Utility Functions
const formatPercent = (val) => `${val >= 0 ? "+" : ""}${val.toFixed(2)}%`;
const formatNumber = (val) =>
  val?.toLocaleString(undefined, { maximumFractionDigits: 2 }) ?? "-";

// Custom Tooltip Component
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip font-mono text-sm">
      <p className="text-indigo-300 mb-2">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color }}>
          {entry.name}: {formatNumber(entry.value)}
        </p>
      ))}
    </div>
  );
};

// Stat Card Component
const StatCard = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  color = "indigo",
}) => {
  const colorClasses = {
    indigo: "from-indigo-500/20 to-purple-500/20 border-indigo-500/30",
    green: "from-emerald-500/20 to-cyan-500/20 border-emerald-500/30",
    red: "from-red-500/20 to-orange-500/20 border-red-500/30",
    blue: "from-blue-500/20 to-cyan-500/20 border-blue-500/30",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`stat-card glass rounded-2xl p-5 bg-gradient-to-br ${colorClasses[color]}`}
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-indigo-300/70 text-sm font-medium">{title}</span>
        {Icon && <Icon className="w-5 h-5 text-indigo-400/50" />}
      </div>
      <div className="flex items-end gap-3">
        <span className="text-3xl font-bold text-white font-mono">{value}</span>
        {trend !== undefined && (
          <span
            className={`flex items-center text-sm ${
              trend >= 0 ? "text-emerald-400" : "text-red-400"
            }`}
          >
            {trend >= 0 ? (
              <TrendingUp className="w-4 h-4 mr-1" />
            ) : (
              <TrendingDown className="w-4 h-4 mr-1" />
            )}
            {formatPercent(trend)}
          </span>
        )}
      </div>
      {subtitle && (
        <p className="text-indigo-300/50 text-xs mt-2">{subtitle}</p>
      )}
    </motion.div>
  );
};

// Ticker Badge Component
const TickerBadge = ({ ticker, weight }) => (
  <div
    className="px-3 py-1.5 rounded-lg font-mono text-sm flex items-center gap-2
      bg-gradient-to-r from-indigo-500/20 to-purple-500/20 border border-indigo-500/30"
  >
    <span className="text-white font-medium">{ticker}</span>
    {weight !== undefined && (
      <span className="text-indigo-300/70">{(weight * 100).toFixed(1)}%</span>
    )}
  </div>
);

// Loading Spinner Component
const LoadingSpinner = ({ message = "Loading..." }) => (
  <div className="flex flex-col items-center justify-center py-12">
    <div className="spinner mb-4"></div>
    <p className="text-indigo-300/70 text-sm">{message}</p>
  </div>
);

// Progress Steps Component
const ProgressSteps = ({ currentStep, steps }) => (
  <div className="flex items-center gap-2 mb-6">
    {steps.map((step, i) => (
      <React.Fragment key={step}>
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm
          ${
            i <= currentStep
              ? "bg-indigo-500/30 text-indigo-200"
              : "bg-midnight-900/50 text-indigo-400/50"
          }`}
        >
          <span
            className={`w-5 h-5 rounded-full flex items-center justify-center text-xs
            ${
              i < currentStep
                ? "bg-emerald-500 text-white"
                : i === currentStep
                ? "bg-indigo-500 text-white"
                : "bg-midnight-800"
            }`}
          >
            {i < currentStep ? "✓" : i + 1}
          </span>
          <span className="hidden sm:inline">{step}</span>
        </div>
        {i < steps.length - 1 && (
          <ChevronRight className="w-4 h-4 text-indigo-500/30" />
        )}
      </React.Fragment>
    ))}
  </div>
);

// Main App Component
function App() {
  // State
  const [selectedBasket, setSelectedBasket] = useState("tech_leaders");
  const [customTickers, setCustomTickers] = useState("");
  const [tickers, setTickers] = useState(BASKET_OPTIONS.tech_leaders.tickers);
  const [objective, setObjective] = useState("sharpe");
  const [weightMethod, setWeightMethod] = useState("mean_variance");
  const [iterations, setIterations] = useState(50);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [backtestData, setBacktestData] = useState(null);
  const [weights, setWeights] = useState({});
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState(null);
  const [priceData, setPriceData] = useState(null);
  const [isLoadingData, setIsLoadingData] = useState(false);

  // Track what config was used for loaded data
  const [loadedConfig, setLoadedConfig] = useState(null);

  // Date range - default to 2 years of data
  const getDefaultDates = () => {
    const end = new Date();
    const start = new Date();
    start.setFullYear(start.getFullYear() - 2);
    return {
      start: start.toISOString().split("T")[0],
      end: end.toISOString().split("T")[0],
    };
  };
  const defaultDates = getDefaultDates();
  const [startDate, setStartDate] = useState(defaultDates.start);
  const [endDate, setEndDate] = useState(defaultDates.end);

  // Check if current config matches loaded data config
  const currentConfig = JSON.stringify({
    tickers: [...tickers].sort(),
    startDate,
    endDate,
  });
  const configChanged = loadedConfig !== currentConfig;
  const dataIsLoaded = priceData !== null && !configChanged;

  // Clear optimization results when config changes
  useEffect(() => {
    if (configChanged && optimizationResult) {
      setOptimizationResult(null);
      setBacktestData(null);
      setWeights({});
      setCurrentStep(0);
    }
  }, [configChanged, optimizationResult]);

  const CHART_COLORS = [
    "#6366f1",
    "#00ff88",
    "#00d4ff",
    "#a855f7",
    "#f97316",
    "#ec4899",
    "#10b981",
    "#f59e0b",
  ];

  // Handle basket selection
  const handleBasketChange = (basket) => {
    setSelectedBasket(basket);
    if (basket !== "custom") {
      setTickers(BASKET_OPTIONS[basket].tickers);
      setCustomTickers("");
    }
  };

  // Handle custom tickers
  const handleCustomTickers = () => {
    const parsed = customTickers
      .toUpperCase()
      .split(/[,\s]+/)
      .filter((t) => t.length > 0 && t.length <= 5);
    if (parsed.length >= 2) {
      setTickers(parsed);
      setSelectedBasket("custom");
    }
  };

  // Fetch price data
  const fetchPriceData = async () => {
    setIsLoadingData(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/data/fetch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers,
          start_date: startDate,
          end_date: endDate,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setPriceData(data);
        setLoadedConfig(
          JSON.stringify({ tickers: [...tickers].sort(), startDate, endDate })
        );
        setCurrentStep(1);
      } else {
        setError(data.detail || "Failed to fetch data");
      }
    } catch (err) {
      setError("Network error: " + err.message);
    } finally {
      setIsLoadingData(false);
    }
  };

  // Run optimization
  const runOptimization = async () => {
    setIsOptimizing(true);
    setError(null);
    setCurrentStep(2);

    try {
      const res = await fetch(`${API_BASE}/optimize/quick`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers,
          start_date: startDate,
          end_date: endDate,
          objective,
          n_iterations: iterations,
          optimization_method: "bayesian",
          weight_method: weightMethod,
          initial_capital: 100000,
        }),
      });

      const data = await res.json();

      if (res.ok) {
        setOptimizationResult(data.result);
        setBacktestData(data.result.backtest_data);

        // Set weights
        const weightObj = {};
        data.result.initial_weights.forEach((w, i) => {
          weightObj[tickers[i]] = w;
        });
        setWeights(weightObj);
        setCurrentStep(3);
      } else {
        setError(data.detail || "Optimization failed");
      }
    } catch (err) {
      setError("Network error: " + err.message);
    } finally {
      setIsOptimizing(false);
    }
  };

  // Chart data preparation
  const getPortfolioChartData = () => {
    if (!backtestData) return [];
    return backtestData.dates.map((date, i) => ({
      date: date.slice(5), // MM-DD format
      value: backtestData.portfolio_values[i],
      returns: backtestData.cumulative_returns[i],
      drawdown: backtestData.drawdown[i],
    }));
  };

  const getConvergenceData = () => {
    if (!optimizationResult?.optimization_result?.convergence_data) return [];
    // Filter out penalty values (1000) and skip first iteration
    const data = optimizationResult.optimization_result.convergence_data;
    return data
      .map((val, i) => ({ iteration: i + 1, score: val }))
      .filter((item) => item.score < 100 && item.iteration > 1); // Skip penalties and first iteration
  };

  const getWeightsPieData = () => {
    return Object.entries(weights).map(([ticker, weight]) => ({
      name: ticker,
      value: weight * 100,
    }));
  };

  const metrics = optimizationResult?.final_metrics;

  return (
    <div className="min-h-screen gradient-animate grid-bg">
      {/* Header */}
      <header className="border-b border-indigo-500/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">
                  Basket Trading Optimizer
                </h1>
                <p className="text-indigo-300/60 text-sm">
                  Powered by Bayesian ML
                </p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-emerald-400 text-sm">
                <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-live"></span>
                Live
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Progress Steps */}
        <ProgressSteps
          currentStep={currentStep}
          steps={["Select Basket", "Load Data", "Optimize", "Results"]}
        />

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 p-4 rounded-xl bg-red-500/20 border border-red-500/30 flex items-center gap-3"
            >
              <AlertCircle className="w-5 h-5 text-red-400" />
              <span className="text-red-200">{error}</span>
              <button
                onClick={() => setError(null)}
                className="ml-auto text-red-400 hover:text-red-300"
              >
                ×
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Configuration Panel */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-3xl p-6 mb-8"
        >
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5 text-indigo-400" />
            Configuration
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Basket Selection */}
            <div>
              <label className="block text-indigo-300/70 text-sm mb-2">
                Select Basket
              </label>
              <div className="flex flex-wrap gap-2 mb-4">
                {Object.entries(BASKET_OPTIONS).map(([key, { name }]) => (
                  <button
                    key={key}
                    onClick={() => handleBasketChange(key)}
                    className={`px-4 py-2 rounded-xl text-sm font-medium transition-all
                      ${
                        selectedBasket === key
                          ? "bg-indigo-500 text-white"
                          : "bg-midnight-900/50 text-indigo-300 hover:bg-indigo-500/20"
                      }`}
                  >
                    {name}
                  </button>
                ))}
              </div>

              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Or enter tickers: AAPL, MSFT, GOOGL..."
                  value={customTickers}
                  onChange={(e) => setCustomTickers(e.target.value)}
                  className="flex-1 px-4 py-2 rounded-xl bg-midnight-950/50 border border-indigo-500/20 
                    text-white placeholder-indigo-300/30 focus:outline-none input-glow font-mono text-sm"
                />
                <button
                  onClick={handleCustomTickers}
                  className="px-4 py-2 rounded-xl bg-indigo-500/20 text-indigo-200 hover:bg-indigo-500/30"
                >
                  Set
                </button>
              </div>
            </div>

            {/* Optimization Settings */}
            <div>
              <label className="block text-indigo-300/70 text-sm mb-2">
                Optimization Objective
              </label>
              <select
                value={objective}
                onChange={(e) => setObjective(e.target.value)}
                className="w-full px-4 py-2 rounded-xl bg-midnight-950/50 border border-indigo-500/20 
                  text-white focus:outline-none input-glow mb-4"
              >
                <option value="sharpe">Sharpe Ratio</option>
                <option value="sortino">Sortino Ratio</option>
                <option value="calmar">Calmar Ratio</option>
                <option value="return">Annualized Return</option>
                <option value="risk_adjusted">Risk-Adjusted</option>
              </select>

              <label className="block text-indigo-300/70 text-sm mb-2">
                Weight Method
              </label>
              <select
                value={weightMethod}
                onChange={(e) => setWeightMethod(e.target.value)}
                className="w-full px-4 py-2 rounded-xl bg-midnight-950/50 border border-indigo-500/20 
                  text-white focus:outline-none input-glow mb-4"
              >
                <option value="mean_variance">Mean-Variance (Markowitz)</option>
                <option value="risk_parity">Risk Parity</option>
                <option value="momentum">Momentum</option>
                <option value="min_variance">Minimum Variance</option>
              </select>

              <label className="block text-indigo-300/70 text-sm mb-2">
                Iterations: {iterations}
              </label>
              <input
                type="range"
                min="10"
                max="100"
                value={iterations}
                onChange={(e) => setIterations(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
            </div>

            {/* Selected Tickers & Date Range */}
            <div>
              <label className="block text-indigo-300/70 text-sm mb-2">
                Date Range
              </label>
              <div className="flex gap-2 mb-4">
                <div className="flex-1">
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl bg-midnight-950/50 border border-indigo-500/20 
                      text-white focus:outline-none input-glow text-sm"
                  />
                  <span className="text-indigo-300/50 text-xs">Start</span>
                </div>
                <div className="flex-1">
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl bg-midnight-950/50 border border-indigo-500/20 
                      text-white focus:outline-none input-glow text-sm"
                  />
                  <span className="text-indigo-300/50 text-xs">End</span>
                </div>
              </div>

              <label className="block text-indigo-300/70 text-sm mb-2">
                Selected Tickers ({tickers.length})
              </label>
              <div className="flex flex-wrap gap-2 mb-4">
                {tickers.map((ticker) => (
                  <TickerBadge
                    key={ticker}
                    ticker={ticker}
                    weight={weights[ticker]}
                  />
                ))}
              </div>

              <div className="flex gap-2">
                <button
                  onClick={fetchPriceData}
                  disabled={
                    tickers.length < 2 || isLoadingData || !configChanged
                  }
                  className="flex-1 btn-primary px-4 py-2 rounded-xl text-white font-medium 
                    disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isLoadingData ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    <>
                      <Activity className="w-4 h-4" />
                      {dataIsLoaded ? "Data Loaded ✓" : "Load Data"}
                    </>
                  )}
                </button>
                <button
                  onClick={runOptimization}
                  disabled={isOptimizing || !dataIsLoaded}
                  className="flex-1 btn-primary px-4 py-2 rounded-xl text-white font-medium 
                    disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isOptimizing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Optimizing...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      Optimize
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Data Loaded Success */}
        {dataIsLoaded && !optimizationResult && !isOptimizing && (
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-3xl p-6 mb-8"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">
                  Data Loaded Successfully
                </h3>
                <p className="text-indigo-300/60 text-sm">
                  {priceData.date_range?.trading_days} trading days from{" "}
                  {priceData.date_range?.start} to {priceData.date_range?.end}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
              {priceData.statistics &&
                Object.entries(priceData.statistics).map(([ticker, stats]) => (
                  <div
                    key={ticker}
                    className="bg-midnight-950/50 rounded-xl p-3 border border-indigo-500/10"
                  >
                    <p className="text-white font-mono font-medium text-sm mb-1">
                      {ticker}
                    </p>
                    <p className="text-lg font-bold text-white">
                      ${stats.current_price}
                    </p>
                    <p
                      className={`text-xs ${
                        stats.total_return >= 0
                          ? "text-emerald-400"
                          : "text-red-400"
                      }`}
                    >
                      {stats.total_return >= 0 ? "+" : ""}
                      {stats.total_return}%
                    </p>
                  </div>
                ))}
            </div>

            <p className="text-indigo-300/50 text-sm mt-4 text-center">
              Click <strong>Optimize</strong> to run Bayesian optimization on
              this basket
            </p>
          </motion.section>
        )}

        {/* Loading State */}
        {isOptimizing && (
          <motion.section
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass rounded-3xl p-8 mb-8 text-center"
          >
            <LoadingSpinner message="Running Bayesian optimization..." />
            <p className="text-indigo-300/50 text-sm mt-4">
              Using Gaussian Process regression with Expected Improvement
              acquisition
            </p>
          </motion.section>
        )}

        {/* Results */}
        {optimizationResult && !isOptimizing && (
          <>
            {/* Metrics Grid */}
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
            >
              <StatCard
                title="Sharpe Ratio"
                value={metrics?.sharpe_ratio ?? "-"}
                subtitle="Risk-adjusted return"
                icon={Target}
                color="indigo"
              />
              <StatCard
                title="Annual Return"
                value={`${metrics?.annualized_return ?? 0}%`}
                subtitle="Annualized"
                icon={TrendingUp}
                trend={metrics?.annualized_return}
                color={metrics?.annualized_return >= 0 ? "green" : "red"}
              />
              <StatCard
                title="Max Drawdown"
                value={`${metrics?.max_drawdown ?? 0}%`}
                subtitle="Peak to trough"
                icon={TrendingDown}
                color="red"
              />
              <StatCard
                title="Volatility"
                value={`${metrics?.volatility ?? 0}%`}
                subtitle="Annualized std dev"
                icon={Activity}
                color="blue"
              />
            </motion.section>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* Portfolio Performance */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="glass rounded-3xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-indigo-400" />
                  Portfolio Performance
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={getPortfolioChartData()}>
                    <defs>
                      <linearGradient
                        id="colorReturns"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor="#6366f1"
                          stopOpacity={0.4}
                        />
                        <stop
                          offset="95%"
                          stopColor="#6366f1"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#312e81" />
                    <XAxis
                      dataKey="date"
                      stroke="#818cf8"
                      tick={{ fill: "#818cf8", fontSize: 11 }}
                    />
                    <YAxis
                      stroke="#818cf8"
                      tick={{ fill: "#818cf8", fontSize: 11 }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="returns"
                      stroke="#6366f1"
                      strokeWidth={2}
                      fill="url(#colorReturns)"
                      name="Cumulative Return (%)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </motion.div>

              {/* Drawdown Chart */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass rounded-3xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <TrendingDown className="w-5 h-5 text-red-400" />
                  Drawdown
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={getPortfolioChartData()}>
                    <defs>
                      <linearGradient
                        id="colorDrawdown"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor="#ef4444"
                          stopOpacity={0.4}
                        />
                        <stop
                          offset="95%"
                          stopColor="#ef4444"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#312e81" />
                    <XAxis
                      dataKey="date"
                      stroke="#818cf8"
                      tick={{ fill: "#818cf8", fontSize: 11 }}
                    />
                    <YAxis
                      stroke="#818cf8"
                      tick={{ fill: "#818cf8", fontSize: 11 }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="drawdown"
                      stroke="#ef4444"
                      strokeWidth={2}
                      fill="url(#colorDrawdown)"
                      name="Drawdown (%)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </motion.div>

              {/* Optimization Convergence */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="glass rounded-3xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <Zap className="w-5 h-5 text-cyber-green" />
                  Bayesian Optimization Convergence
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={getConvergenceData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#312e81" />
                    <XAxis
                      dataKey="iteration"
                      stroke="#818cf8"
                      tick={{ fill: "#818cf8", fontSize: 11 }}
                    />
                    <YAxis
                      stroke="#818cf8"
                      tick={{ fill: "#818cf8", fontSize: 11 }}
                      domain={["dataMin - 0.1", "dataMax + 0.1"]}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Line
                      type="monotone"
                      dataKey="score"
                      stroke="#00ff88"
                      strokeWidth={2}
                      dot={true}
                      name="Best Score"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </motion.div>

              {/* Portfolio Weights */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="glass rounded-3xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <PieChartIcon className="w-5 h-5 text-purple-400" />
                  Portfolio Allocation
                </h3>
                <div className="flex items-center gap-4">
                  <ResponsiveContainer width="50%" height={250}>
                    <PieChart>
                      <Pie
                        data={getWeightsPieData()}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        innerRadius={50}
                        outerRadius={80}
                        paddingAngle={2}
                      >
                        {getWeightsPieData().map((_, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={CHART_COLORS[index % CHART_COLORS.length]}
                          />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="flex-1 space-y-2">
                    {Object.entries(weights).map(([ticker, weight], i) => (
                      <div key={ticker} className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{
                            backgroundColor:
                              CHART_COLORS[i % CHART_COLORS.length],
                          }}
                        />
                        <span className="font-mono text-sm text-indigo-200">
                          {ticker}
                        </span>
                        <span className="ml-auto font-mono text-sm text-indigo-300/70">
                          {(weight * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Optimized Parameters */}
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="glass rounded-3xl p-6 mb-8"
            >
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                Optimized Strategy Parameters
              </h3>

              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {optimizationResult?.optimization_result?.best_params &&
                  Object.entries(
                    optimizationResult.optimization_result.best_params
                  ).map(([key, value]) => (
                    <div
                      key={key}
                      className="bg-midnight-950/50 rounded-xl p-3 border border-indigo-500/10"
                    >
                      <p className="text-indigo-300/50 text-xs mb-1">
                        {key.replace(/_/g, " ")}
                      </p>
                      <p className="text-white font-mono text-sm">
                        {typeof value === "number"
                          ? value.toFixed(3)
                          : String(value)}
                      </p>
                    </div>
                  ))}
              </div>

              <div className="mt-4 flex items-center gap-4 text-sm text-indigo-300/50">
                <span className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  Computation time:{" "}
                  {optimizationResult?.optimization_result?.computation_time?.toFixed(
                    1
                  )}
                  s
                </span>
                <span>•</span>
                <span>
                  Iterations:{" "}
                  {optimizationResult?.optimization_result?.total_iterations}
                </span>
                <span>•</span>
                <span>
                  Best {objective}:{" "}
                  {optimizationResult?.optimization_result?.best_score?.toFixed(
                    4
                  )}
                </span>
              </div>
            </motion.section>

            {/* Additional Metrics */}
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="glass rounded-3xl p-6"
            >
              <h3 className="text-lg font-semibold text-white mb-4">
                Additional Metrics
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                <div className="text-center p-4 bg-midnight-950/30 rounded-xl">
                  <p className="text-indigo-300/50 text-xs mb-1">
                    Sortino Ratio
                  </p>
                  <p className="text-xl font-bold text-white">
                    {metrics?.sortino_ratio ?? "-"}
                  </p>
                </div>
                <div className="text-center p-4 bg-midnight-950/30 rounded-xl">
                  <p className="text-indigo-300/50 text-xs mb-1">
                    Calmar Ratio
                  </p>
                  <p className="text-xl font-bold text-white">
                    {metrics?.calmar_ratio ?? "-"}
                  </p>
                </div>
                <div className="text-center p-4 bg-midnight-950/30 rounded-xl">
                  <p className="text-indigo-300/50 text-xs mb-1">Win Rate</p>
                  <p className="text-xl font-bold text-white">
                    {metrics?.win_rate ?? "-"}%
                  </p>
                </div>
                <div className="text-center p-4 bg-midnight-950/30 rounded-xl">
                  <p className="text-indigo-300/50 text-xs mb-1">
                    Profit Factor
                  </p>
                  <p className="text-xl font-bold text-white">
                    {metrics?.profit_factor ?? "-"}
                  </p>
                </div>
                <div className="text-center p-4 bg-midnight-950/30 rounded-xl">
                  <p className="text-indigo-300/50 text-xs mb-1">Beta</p>
                  <p className="text-xl font-bold text-white">
                    {metrics?.beta ?? "-"}
                  </p>
                </div>
                <div className="text-center p-4 bg-midnight-950/30 rounded-xl">
                  <p className="text-indigo-300/50 text-xs mb-1">Alpha</p>
                  <p className="text-xl font-bold text-white">
                    {metrics?.alpha ?? "-"}%
                  </p>
                </div>
              </div>
            </motion.section>
          </>
        )}

        {/* Initial State */}
        {!optimizationResult && !isOptimizing && (
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-3xl p-12 text-center"
          >
            <div
              className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-indigo-500/30 to-purple-500/30 
              flex items-center justify-center border border-indigo-500/30"
            >
              <Sparkles className="w-10 h-10 text-indigo-400" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-3">
              Bayesian Portfolio Optimization
            </h2>
            <p className="text-indigo-300/60 max-w-lg mx-auto mb-6">
              Select a stock basket and run optimization to find the best
              trading strategy parameters using Gaussian Process-based Bayesian
              optimization.
            </p>
          </motion.section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-indigo-500/20 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6 text-center text-indigo-300/40 text-sm">
          <p>
            Basket Trading Optimizer • Bayesian ML-powered strategy optimization
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
