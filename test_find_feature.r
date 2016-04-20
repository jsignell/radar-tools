
file <- read.csv("C:/Users/Julia/Downloads/Charlotte_box_radar_2006_08.csv", 
                 skip=2860, nrows=13, header=FALSE)
locs <- read.csv("C:/Users/Julia/Downloads/Charlotte_box_latlon_big.csv", 
                 header=FALSE)
ll <- cbind(locs$V3, locs$V2)
colnames(ll) <- c('lon','lat')
f <- list()

i=12
x <- matrix(as.numeric(file[i,6:19605]), nrow=140)
xhat <- matrix(as.numeric(file[i+1,6:19605]), nrow=140)

par( mfrow=c(1,2))
quilt.plot(ll, x, nx=140, ny=140)


hold <- make.SpatialVx(x,xhat, loc=ll, projection=TRUE, map=TRUE)
look <- FeatureFinder(hold, smoothpar=3, thresh=10, min.size=400)
plot(look,type = 'obs')

m <- centmatch(look, criteria=3, const=15)

k <-length(f)
f[k+1:(k+length(FeatureMatchAnalyzer(m)))] <-FeatureMatchAnalyzer(m)

