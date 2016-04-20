require('SpatialVx')

data(ExampleSpatialVxSet)
x <- ExampleSpatialVxSet$vx
xhat <- ExampleSpatialVxSet$fcst
par(mfrow=c(1,2))
image.plot(x, col=c("gray",tim.colors(64)))
image.plot(xhat, col=c("gray",tim.colors(64)))

data(pert000)
data(pert004)
data(ICPg240Locs)
hold <- make.SpatialVx(pert000, pert004,
                       loc=ICPg240Locs, projection=TRUE, map=TRUE,
                       loc.byrow = TRUE,
                       field.type="Precipitation", units="mm/h",
                       data.name=c("Perturbed ICP Cases", "pert000", "pert004"))
look <- FeatureFinder(hold, smoothpar=10.5)

plot(look)

x <- y <- matrix(0, 100, 100)
x[2:3,c(3:6, 8:10)] <- 1
y[c(4:7, 9:10),c(7:9, 11:12)] <- 1
x[30:50,45:65] <- 1
y[c(22:24, 99:100),c(50:52, 99:100)] <- 1
hold <- make.SpatialVx(x, y, field.type="contrived", units="none",
                       data.name=c("Example", "x", "y"))
par( mfrow=c(1,2))
image.plot(x)
image.plot(y)