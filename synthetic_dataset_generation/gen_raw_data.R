library(tspgen)

parse_args <- function() {
  defaults <- list(sizes = "25,50,100", n_graphs = "3000",
                   iters = "50", seed = "42", out_dir = "raw_gen_dataset")
  args <- commandArgs(trailingOnly = TRUE)
  for (a in args) {
    kv <- sub("^--", "", a)
    key <- sub("=.*$", "", kv)
    val <- sub("^[^=]*=", "", kv)
    if (key %in% names(defaults)) defaults[[key]] <- val
  }
  defaults$sizes    <- as.integer(strsplit(defaults$sizes, ",")[[1]])
  defaults$n_graphs <- as.integer(defaults$n_graphs)
  defaults$iters    <- as.integer(defaults$iters)
  defaults$seed     <- as.integer(defaults$seed)
  defaults
}

opt <- parse_args()
set.seed(opt$seed)

collection <- init()
collection <- addMutator(collection, "doExplosionMutation")
collection <- addMutator(collection, "doClusterMutation", pm = 0.2)
collection <- addMutator(collection, "doAxisProjectionMutation", pm = 0.3)
collection <- addMutator(collection, "doGridMutation", box.min = 0.3, box.max = 0.5)
collection <- setProbabilities(collection, probs = c(0.3, 0.2, 0.3, 0.2))

generate_one_graph <- function(n_customers, graph_id, collection, iters) {
  x <- build(n = as.integer(n_customers), iters = as.integer(iters),
             collection = collection, bound.handling = "uniform")
  coords <- as.data.frame(x$coordinates)
  colnames(coords) <- c("x", "y")
  coords$graph_id <- graph_id
  coords$node_id <- seq_len(nrow(coords))
  coords[, c("graph_id", "node_id", "x", "y")]
}

generate_many_graphs <- function(n_customers, n_graphs, collection, iters, out_dir) {
  all_graphs <- vector("list", n_graphs)
  for (gid in seq_len(n_graphs)) {
    if (gid %% 100 == 0)
      cat("Generating n =", n_customers, ", graph", gid, "/", n_graphs, "\n")
    all_graphs[[gid]] <- generate_one_graph(n_customers, gid, collection, iters)
  }
  big_df <- do.call(rbind, all_graphs)

  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  out_file <- file.path(out_dir, paste0("tsp_n", n_customers, "_", n_graphs, "graphs.csv"))
  write.csv(big_df, out_file, row.names = FALSE)
  cat("Saved:", out_file, " (rows:", nrow(big_df), ")\n\n")
  invisible(big_df)
}

cat("sizes =", paste(opt$sizes, collapse = ","),
    "| n_graphs =", opt$n_graphs, "| iters =", opt$iters,
    "| seed =", opt$seed, "| out_dir =", opt$out_dir, "\n\n")

for (n in opt$sizes) {
  generate_many_graphs(n, opt$n_graphs, collection, opt$iters, opt$out_dir)
}
